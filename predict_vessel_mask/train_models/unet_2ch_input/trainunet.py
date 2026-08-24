import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from skimage.filters import frangi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    ROOT / "data" / "augmented_training_dataset_3x"
)
DEFAULT_FRANGI_ROOT = (
    ROOT / "data" / "augmented_training_dataset_3x_frangi_sigma_2_20" / "frangi"
)
DEFAULT_PA_ROOT = (
    ROOT
    / "data"
    / "Photoacoustic vascular image dataset"
    / "Photoacoustic vascular image dataset"
    / "ear_vessel_dataset"
    / "ear_vessel_data"
)
DEFAULT_OUTPUT_DIR = ROOT / "unet_runs" / "rsom_pa_unet_topology"
DEFAULT_INITIAL_CHECKPOINT = ""


@dataclass(frozen=True)
class Sample:
    file_name: str
    image_path: Path
    mask_path: Path
    source: str
    source_id: str
    variant: str
    frangi_path: Path | None = None


class ImageMaskDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 256,
        use_frangi: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.use_frangi = use_frangi

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_grayscale(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            array = np.asarray(image).copy()

        if array.ndim != 2:
            raise ValueError(f"Expected grayscale image: {path}")

        return array

    @staticmethod
    def _normalize_01(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)

        minimum = float(array.min())
        maximum = float(array.max())

        if maximum - minimum < 1e-8:
            return np.zeros_like(array, dtype=np.float32)

        return (array - minimum) / (maximum - minimum)

    def _pad(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape

        if height > self.image_size or width > self.image_size:
            raise ValueError(
                f"{array.shape} is larger than "
                f"{(self.image_size, self.image_size)}"
            )

        top = (self.image_size - height) // 2
        bottom = self.image_size - height - top
        left = (self.image_size - width) // 2
        right = self.image_size - width - left

        return np.pad(
            array,
            ((top, bottom), (left, right)),
            constant_values=0,
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        raw_image = self._read_grayscale(sample.image_path)
        raw_mask = self._read_grayscale(sample.mask_path)

        if raw_image.shape != raw_mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {sample.file_name}: "
                f"{raw_image.shape} != {raw_mask.shape}"
            )

        if not np.issubdtype(raw_image.dtype, np.integer):
            raise ValueError(
                f"Expected integer image, got {raw_image.dtype}: "
                f"{sample.image_path}"
            )

        image_max = float(np.iinfo(raw_image.dtype).max)
        original = raw_image.astype(np.float32) / image_max

        mask = (raw_mask > 0).astype(np.float32)

        if self.use_frangi:
            if sample.frangi_path is not None:
                raw_frangi = self._read_grayscale(sample.frangi_path)
                if raw_frangi.shape != raw_image.shape:
                    raise ValueError(
                        f"Image/Frangi mismatch for {sample.file_name}: "
                        f"{raw_image.shape} != {raw_frangi.shape}"
                    )
                if not np.issubdtype(raw_frangi.dtype, np.integer):
                    raise ValueError(
                        f"Expected integer Frangi image: {sample.frangi_path}"
                    )
                frangi_max = float(np.iinfo(raw_frangi.dtype).max)
                vesselness = raw_frangi.astype(np.float32) / frangi_max
            else:
                # The official PA test set is outside the prepared training
                # dataset. Compute its second channel once during final test.
                vesselness = frangi(
                    original,
                    sigmas=range(2, 21),
                    black_ridges=False,
                    mode="reflect",
                )
                scale = float(np.percentile(vesselness, 99.9))
                if np.isfinite(scale) and scale > 0:
                    vesselness = np.clip(vesselness / scale, 0.0, 1.0)
                else:
                    vesselness = np.zeros_like(original, dtype=np.float32)

        original = self._pad(original)
        mask = self._pad(mask)

        if self.use_frangi:
            vesselness = self._pad(vesselness)

            # [2, H, W]
            image = np.stack(
                [original, vesselness],
                axis=0,
            ).astype(np.float32)
        else:
            # 기존 1채널 모드
            image = np.expand_dims(
                original,
                axis=0,
            ).astype(np.float32)

        mask = np.expand_dims(
            mask,
            axis=0,
        ).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor


def normalization_groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = normalization_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32
    ) -> None:
        super().__init__()
        features = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.encoder1 = ConvBlock(in_channels, features[0])
        self.encoder2 = ConvBlock(features[0], features[1])
        self.encoder3 = ConvBlock(features[1], features[2])
        self.encoder4 = ConvBlock(features[2], features[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(
            features[3] * 2, features[3], kernel_size=2, stride=2
        )
        self.decoder4 = ConvBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(
            features[3], features[2], kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(
            features[2], features[1], kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(
            features[1], features[0], kernel_size=2, stride=2
        )
        self.decoder1 = ConvBlock(features[0] * 2, features[0])
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.decoder4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.output(dec1)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(
            dim=dimensions
        )
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        ).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    eroded_h = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    eroded_w = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_h, eroded_w)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Differentiable skeleton approximation used by soft-clDice."""
    image = image.float()
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class TopologyAwareLoss(nn.Module):
    """BCE+Dice with a soft-clDice continuity constraint for thin vessels."""

    def __init__(self, topology_weight: float = 0.2) -> None:
        super().__init__()
        self.region_loss = BCEDiceLoss(bce_weight=0.5)
        self.topology_weight = topology_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        region = self.region_loss(logits, targets)
        probabilities = torch.sigmoid(logits)
        predicted_skeleton = soft_skeletonize(probabilities)
        target_skeleton = soft_skeletonize(targets)
        epsilon = 1e-6
        topology_precision = (
            (predicted_skeleton * targets).sum(dim=(1, 2, 3)) + epsilon
        ) / (predicted_skeleton.sum(dim=(1, 2, 3)) + epsilon)
        topology_sensitivity = (
            (target_skeleton * probabilities).sum(dim=(1, 2, 3)) + epsilon
        ) / (target_skeleton.sum(dim=(1, 2, 3)) + epsilon)
        soft_cldice = (
            2.0 * topology_precision * topology_sensitivity
            / (topology_precision + topology_sensitivity + epsilon)
        ).mean()
        return region + self.topology_weight * (1.0 - soft_cldice)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pa_group(source_id: str) -> str:
    parts = source_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected PA source_id: {source_id}")
    return "_".join(parts[:2])


def load_augmented_splits(
    data_root: Path,
    frangi_root: Path,
    pa_val_group: str,
    rsom_val_start: int,
    rsom_val_end: int,
    rsom_val_buffer: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    manifest_path = data_root / "manifest.csv"
    train_samples: list[Sample] = []
    pa_val_samples: list[Sample] = []
    rsom_val_samples: list[Sample] = []

    with manifest_path.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            file_name = row["file_name"]
            sample = Sample(
                file_name=file_name,
                image_path=data_root / "images" / file_name,
                mask_path=data_root / "masks" / file_name,
                source=row["source"],
                source_id=row["source_id"],
                variant=row["variant"],
                frangi_path=frangi_root / file_name,
            )

            if (
                not sample.image_path.is_file()
                or not sample.mask_path.is_file()
                or sample.frangi_path is None
                or not sample.frangi_path.is_file()
            ):
                raise FileNotFoundError(
                    f"Missing original/Frangi/mask triplet: {file_name}"
                )

            if sample.source == "photoacoustic_train":
                if pa_group(sample.source_id) == pa_val_group:
                    if sample.variant == "original":
                        pa_val_samples.append(sample)
                else:
                    train_samples.append(sample)
                continue

            if sample.source == "rsom":
                slice_index = int(sample.source_id)
                if rsom_val_start <= slice_index <= rsom_val_end:
                    if sample.variant == "original":
                        rsom_val_samples.append(sample)
                    continue

                buffer_start = max(0, rsom_val_start - rsom_val_buffer)
                buffer_end = min(399, rsom_val_end + rsom_val_buffer)
                if buffer_start <= slice_index <= buffer_end:
                    continue

                train_samples.append(sample)
                continue

            raise ValueError(f"Unknown source: {sample.source}")

    return train_samples, pa_val_samples, rsom_val_samples


def load_pa_test_samples(pa_root: Path) -> list[Sample]:
    image_root = pa_root / "test" / "image"
    mask_root = pa_root / "test" / "groundtruth"
    samples: list[Sample] = []

    for image_path in sorted(image_root.rglob("*.png")):
        relative_path = image_path.relative_to(image_root)
        image_group = relative_path.parts[0]
        mask_group = image_group.replace("patch_", "patch")
        mask_path = mask_root / mask_group / relative_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing PA test mask: {mask_path}")

        source_id = relative_path.with_suffix("").as_posix().replace("/", "_")
        samples.append(
            Sample(
                file_name=f"test_{source_id}.png",
                image_path=image_path,
                mask_path=mask_path,
                source="photoacoustic_test",
                source_id=source_id,
                variant="original",
            )
        )

    return samples


def source_balanced_sampler(
    samples: list[Sample], seed: int
) -> WeightedRandomSampler:
    source_counts = Counter(sample.source for sample in samples)
    weights = [1.0 / source_counts[sample.source] for sample in samples]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )


def make_loader(
    samples: list[Sample],
    batch_size: int,
    workers: int,
    seed: int,
    sampler: WeightedRandomSampler | None = None,
    use_frangi: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ImageMaskDataset(samples, use_frangi=use_frangi),
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def update_confusion(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    true_positive = (predictions & truth).sum().item()
    false_positive = (predictions & ~truth).sum().item()
    false_negative = (~predictions & truth).sum().item()
    return true_positive, false_positive, false_negative


def metrics_from_counts(
    loss_sum: float,
    sample_count: int,
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> dict[str, float]:
    epsilon = 1e-8
    dice = (2.0 * true_positive + epsilon) / (
        2.0 * true_positive + false_positive + false_negative + epsilon
    )
    iou = (true_positive + epsilon) / (
        true_positive + false_positive + false_negative + epsilon
    )
    precision = (true_positive + epsilon) / (
        true_positive + false_positive + epsilon
    )
    recall = (true_positive + epsilon) / (
        true_positive + false_negative + epsilon
    )
    return {
        "loss": loss_sum / max(sample_count, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def binary_image_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Paper-style metrics for one image; callers average over images."""
    prediction = probability >= threshold
    truth = target >= 0.5
    epsilon = 1e-8

    intersection = float(np.logical_and(prediction, truth).sum())
    prediction_sum = float(prediction.sum())
    truth_sum = float(truth.sum())
    dice = (2.0 * intersection + epsilon) / (
        prediction_sum + truth_sum + epsilon
    )
    accuracy = float((prediction == truth).mean())

    predicted_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(truth)
    topology_precision = (
        float(np.logical_and(predicted_skeleton, truth).sum()) + epsilon
    ) / (float(predicted_skeleton.sum()) + epsilon)
    topology_sensitivity = (
        float(np.logical_and(target_skeleton, prediction).sum()) + epsilon
    ) / (float(target_skeleton.sum()) + epsilon)
    cldice = (
        2.0 * topology_precision * topology_sensitivity
        / (topology_precision + topology_sensitivity + epsilon)
    )

    predicted_components = int(label(prediction, connectivity=2).max())
    target_components = int(label(truth, connectivity=2).max())
    predicted_beta1 = predicted_components - int(euler_number(prediction, connectivity=2))
    target_beta1 = target_components - int(euler_number(truth, connectivity=2))

    if prediction.any() and truth.any():
        distance_to_truth = distance_transform_edt(~truth)
        distance_to_prediction = distance_transform_edt(~prediction)
        hausdorff = max(
            float(distance_to_truth[prediction].max()),
            float(distance_to_prediction[truth].max()),
        )
    elif not prediction.any() and not truth.any():
        hausdorff = 0.0
    else:
        hausdorff = float(np.hypot(*prediction.shape))

    if truth.any() and not truth.all():
        auc = float(roc_auc_score(truth.ravel(), probability.ravel()))
    else:
        auc = float("nan")

    return {
        "dice": dice,
        "cldice": cldice,
        "accuracy": accuracy,
        "auc": auc,
        "betti0_error": float(abs(predicted_components - target_components)),
        "betti1_error": float(abs(predicted_beta1 - target_beta1)),
        "hausdorff": hausdorff,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0

    for batch_index, (images, masks) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        tp, fp, fn = update_confusion(logits.detach(), masks)
        true_positive += tp
        false_positive += fp
        false_negative += fn

        if batch_index % 20 == 0:
            print(
                f"  train batch {batch_index}/{len(loader)} "
                f"loss={loss_sum / sample_count:.4f}",
                flush=True,
            )

    return metrics_from_counts(
        loss_sum,
        sample_count,
        true_positive,
        false_positive,
        false_negative,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    per_image_metrics: list[dict[str, float]] = []

    for batch_index, (images, masks) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        targets = masks.float().cpu().numpy()[:, 0]
        per_image_metrics.extend(
            binary_image_metrics(probability, target)
            for probability, target in zip(probabilities, targets)
        )

    metrics = {"loss": loss_sum / max(sample_count, 1)}
    for name in per_image_metrics[0] if per_image_metrics else ():
        values = np.asarray(
            [item[name] for item in per_image_metrics], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        metrics[name] = float(finite.mean()) if finite.size else float("nan")
    return metrics


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in unwrap_model(model).state_dict().items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": cpu_state_dict(model),
            "metrics": metrics,
        },
        path,
    )


def write_split_manifest(
    path: Path,
    train_samples: list[Sample],
    pa_val_samples: list[Sample],
    rsom_val_samples: list[Sample],
    test_samples: list[Sample],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "split",
            "file_name",
            "source",
            "source_id",
            "variant",
            "image_path",
            "mask_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, samples in (
            ("train", train_samples),
            ("val_pa", pa_val_samples),
            ("val_rsom", rsom_val_samples),
            ("test_pa", test_samples),
        ):
            for sample in samples:
                writer.writerow(
                    {
                        "split": split,
                        "file_name": sample.file_name,
                        "source": sample.source,
                        "source_id": sample.source_id,
                        "variant": sample.variant,
                        "image_path": str(sample.image_path),
                        "mask_path": str(sample.mask_path),
                    }
                )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def print_split_summary(
    train_samples: list[Sample],
    pa_val_samples: list[Sample],
    rsom_val_samples: list[Sample],
    test_samples: list[Sample],
) -> None:
    train_counts = Counter(sample.source for sample in train_samples)
    print(f"Train: {len(train_samples)} {dict(train_counts)}")
    print(f"PA validation: {len(pa_val_samples)}")
    print(f"RSOM validation: {len(rsom_val_samples)}")
    print(f"PA official test: {len(test_samples)}")


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{name}={value:.4f}"
        for name, value in metrics.items()
        if name in {
            "loss", "dice", "cldice", "accuracy", "auc",
            "betti0_error", "betti1_error", "hausdorff",
            "iou", "precision", "recall",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D binary U-Net for photoacoustic vessel segmentation."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--frangi-root",
        default=str(DEFAULT_FRANGI_ROOT),
        help="Folder containing precomputed Frangi images matched by file name.",
    )
    parser.add_argument("--pa-root", default=str(DEFAULT_PA_ROOT))
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=str(DEFAULT_INITIAL_CHECKPOINT),
        help="Optional checkpoint. Empty by default to train from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--topology-weight", type=float, default=0.2,
        help="Weight of the differentiable soft-clDice continuity loss.",
    )
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pa-val-group", default="patch_7")
    parser.add_argument("--rsom-val-start", type=int, default=240)
    parser.add_argument("--rsom-val-end", type=int, default=279)
    parser.add_argument("--rsom-val-buffer", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

def load_one_channel_checkpoint_into_two_channel_model(
    model: nn.Module,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    old_state = checkpoint["model_state"]
    new_state = model.state_dict()

    first_conv_key = "encoder1.block.0.weight"

    if first_conv_key not in old_state:
        raise KeyError(
            f"First convolution was not found: {first_conv_key}"
        )

    old_weight = old_state[first_conv_key]

    if old_weight.shape[1] != 1:
        raise ValueError(
            f"Expected one-channel weight, got {old_weight.shape}"
        )

    if new_state[first_conv_key].shape[1] != 2:
        raise ValueError(
            "The new model is not configured for two input channels."
        )

    # 기존 필터를 두 채널에 복사하고 출력 크기를 유지하기 위해 절반씩 배분
    expanded_weight = torch.zeros_like(new_state[first_conv_key])
    expanded_weight[:, 0:1] = old_weight

    old_state[first_conv_key] = expanded_weight

    missing_keys, unexpected_keys = model.load_state_dict(
        old_state,
        strict=False,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    frangi_root = resolve_path(args.frangi_root)
    pa_root = resolve_path(args.pa_root)
    output_dir = resolve_path(args.output_dir)
    initial_checkpoint = (
        resolve_path(args.initial_checkpoint)
        if args.initial_checkpoint
        else None
    )

    train_samples, pa_val_samples, rsom_val_samples = load_augmented_splits(
        data_root=data_root,
        frangi_root=frangi_root,
        pa_val_group=args.pa_val_group,
        rsom_val_start=args.rsom_val_start,
        rsom_val_end=args.rsom_val_end,
        rsom_val_buffer=args.rsom_val_buffer,
    )
    test_samples = load_pa_test_samples(pa_root)
    print_split_summary(
        train_samples, pa_val_samples, rsom_val_samples, test_samples
    )

    if args.dry_run:
        print("Dry run complete.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(
        output_dir / "splits.csv",
        train_samples,
        pa_val_samples,
        rsom_val_samples,
        test_samples,
    )
    configuration = vars(args).copy()
    configuration.update(
        {
            "resolved_data_root": str(data_root),
            "resolved_frangi_root": str(frangi_root),
            "resolved_pa_root": str(pa_root),
            "resolved_output_dir": str(output_dir),
            "resolved_initial_checkpoint": (
                str(initial_checkpoint) if initial_checkpoint else None
            ),
            "input_channels": 2,
            "input_channel_names": ["original", "frangi_sigma_2_20"],
            "paper_reference": "arXiv:2307.08388",
            "topology_note": (
                "soft-clDice is used as a differentiable continuity surrogate; "
                "this is not the unreleased persistent-homology TCLoss"
            ),
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    sampler = source_balanced_sampler(train_samples, args.seed)
    train_loader = make_loader(
        train_samples,
        args.batch_size,
        args.workers,
        args.seed,
        sampler=sampler,
        use_frangi=True,
    )
    pa_val_loader = make_loader(
        pa_val_samples,
        args.batch_size,
        args.workers,
        args.seed + 1,
        use_frangi=True,
    )
    rsom_val_loader = make_loader(
        rsom_val_samples,
        args.batch_size,
        args.workers,
        args.seed + 2,
        use_frangi=True,
    )
    test_loader = make_loader(
        test_samples,
        args.batch_size,
        args.workers,
        args.seed + 3,
        use_frangi=True,
    )

    device = select_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    model: nn.Module = UNet(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    if initial_checkpoint is not None:
        load_one_channel_checkpoint_into_two_channel_model(model, initial_checkpoint)
    else:
        print("Training the two-channel topology-aware U-Net from scratch.")
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    print(f"Device: {device}, AMP: {use_amp}")

    criterion = TopologyAwareLoss(topology_weight=args.topology_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_score = -1.0
    epochs_without_improvement = 0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            use_amp,
            args.max_train_batches,
        )
        pa_val_metrics = evaluate(
            model,
            pa_val_loader,
            criterion,
            device,
            use_amp,
            args.max_eval_batches,
        )
        rsom_val_metrics = evaluate(
            model,
            rsom_val_loader,
            criterion,
            device,
            use_amp,
            args.max_eval_batches,
        )

        validation_score = (
            pa_val_metrics["dice"]
            + pa_val_metrics["cldice"]
            + rsom_val_metrics["dice"]
            + rsom_val_metrics["cldice"]
        ) / 4.0
        validation_loss = (
            pa_val_metrics["loss"] + rsom_val_metrics["loss"]
        ) / 2.0
        scheduler.step(validation_loss)

        row = {
            "epoch": float(epoch),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"pa_val_{key}": value for key, value in pa_val_metrics.items()},
            **{
                f"rsom_val_{key}": value
                for key, value in rsom_val_metrics.items()
            },
            "validation_score": validation_score,
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)

        print(f"  train:    {format_metrics(train_metrics)}")
        print(f"  PA val:   {format_metrics(pa_val_metrics)}")
        print(f"  RSOM val: {format_metrics(rsom_val_metrics)}")
        print(f"  mean validation Dice/clDice={validation_score:.4f}")

        save_checkpoint(
            output_dir / "last_model.pt",
            model,
            epoch,
            row,
        )
        if validation_score > best_score:
            best_score = validation_score
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, epoch, row)
            print("  Saved new best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping:
            print(
                f"Early stopping after {args.early_stopping} "
                "epochs without improvement."
            )
            break

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        args.max_eval_batches,
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, ensure_ascii=False, indent=2)
    print(f"\nPA official test: {format_metrics(test_metrics)}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
