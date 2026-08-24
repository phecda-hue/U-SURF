from __future__ import annotations

"""Train a two-channel U-Net on VessMAP and infer external image folders.

Channel 0 is the normalized grayscale image and channel 1 is its Frangi
vesselness response.  The target and prediction are always a single binary
vessel mask; Frangi is never applied to the target.
"""

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.filters import frangi
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Pair:
    sample_id: str
    image: Path
    mask: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    lo, hi = np.percentile(image, (1.0, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def frangi_channel(image01: np.ndarray) -> np.ndarray:
    response = frangi(
        image01,
        sigmas=range(1, 6),
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=False,
        mode="reflect",
    ).astype(np.float32)
    scale = float(np.percentile(response, 99.5))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(image01, dtype=np.float32)
    return np.clip(response / scale, 0.0, 1.0).astype(np.float32)


def collect_pairs(vessmap_root: Path, annotator: str) -> list[Pair]:
    image_dir = vessmap_root / "images"
    mask_dir = vessmap_root / annotator / "labels"
    images = {
        p.stem: p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    masks = {
        p.stem: p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise FileNotFoundError(f"No image/mask pairs in {vessmap_root}")
    missing = sorted(images.keys() - masks.keys())
    if missing:
        raise RuntimeError(
            f"{annotator} has no labels for {len(missing)} images; "
            f"first missing IDs: {missing[:5]}"
        )
    return [Pair(key, images[key], masks[key]) for key in common]


def split_pairs(
    pairs: list[Pair], seed: int, val_fraction: float, test_fraction: float
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be positive and sum to less than 1")
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    n_val = max(1, round(len(shuffled) * val_fraction))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def training_transform(size: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.90, 1.10), translate_percent=(-0.05, 0.05),
            rotate=(-20, 20), shear=(-5, 5),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
        A.RandomBrightnessContrast(0.12, 0.12, p=0.5),
        A.RandomGamma((85, 115), p=0.35),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.25),
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
    ])


class VessMAPDataset(Dataset):
    def __init__(
        self, pairs: list[Pair], size: int, augment: bool, repeats: int = 1,
        cache: bool = False,
    ) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.repeats = repeats if augment else 1
        self.transform = training_transform(size) if augment else None
        self.cache: list[tuple[torch.Tensor, torch.Tensor, str]] | None = None
        if cache:
            # Freeze the requested augmented copies once. This preserves data
            # augmentation while avoiding an expensive multi-scale Frangi
            # recomputation at every epoch.
            self.cache = [self._make_item(i) for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.pairs) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache is not None:
            return self.cache[index]
        return self._make_item(index)

    def _make_item(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index % len(self.pairs)]
        image = read_gray(pair.image)
        mask = (read_gray(pair.mask) > 0).astype(np.uint8)
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {pair.sample_id}")
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        elif image.shape != (self.size, self.size):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        original = normalize_image(image)
        vesselness = frangi_channel(original)
        inputs = np.stack([original, vesselness]).astype(np.float32)
        target = (mask > 0).astype(np.float32)[None]
        return torch.from_numpy(inputs), torch.from_numpy(target), pair.sample_id


def groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        f = [base, base * 2, base * 4, base * 8]
        self.pool = nn.MaxPool2d(2)
        self.e1, self.e2 = ConvBlock(2, f[0]), ConvBlock(f[0], f[1])
        self.e3, self.e4 = ConvBlock(f[1], f[2]), ConvBlock(f[2], f[3])
        self.b = ConvBlock(f[3], f[3] * 2)
        self.u4 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.d4 = ConvBlock(f[3] * 2, f[3])
        self.u3 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.d3 = ConvBlock(f[2] * 2, f[2])
        self.u2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.d2 = ConvBlock(f[1] * 2, f[1])
        self.u1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.d1 = ConvBlock(f[0] * 2, f[0])
        self.out = nn.Conv2d(f[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        return self.out(self.d1(torch.cat([self.u1(d2), e1], 1)))


class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        dice = (2 * (probs * targets).sum(dims) + 1) / (
            probs.sum(dims) + targets.sum(dims) + 1
        )
        return self.bce(logits, targets) + (1 - dice.mean())


def metrics_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    eps = 1e-7
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); loss_fn = DiceBCELoss()
    total_loss = tp = fp = fn = count = 0.0
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs); total_loss += float(loss_fn(logits, targets)) * len(inputs)
        pred = torch.sigmoid(logits) >= 0.5; truth = targets >= 0.5
        tp += float((pred & truth).sum()); fp += float((pred & ~truth).sum())
        fn += float((~pred & truth).sum()); count += len(inputs)
    return {"loss": total_loss / count, **metrics_from_counts(tp, fp, fn)}


def save_split(output: Path, splits: dict[str, list[Pair]]) -> None:
    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["sample_id", "split", "image", "mask"])
        for split, pairs in splits.items():
            for pair in pairs:
                writer.writerow([pair.sample_id, split, pair.image, pair.mask])


def predict_external(
    model: nn.Module, roots: list[Path], output: Path, device: torch.device, size: int
) -> int:
    model.eval(); count = 0
    for root in roots:
        if not root.exists():
            continue
        destination = output / "predictions" / root.name
        destination.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        for path in files:
            raw = read_gray(path); original_shape = raw.shape
            resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
            image01 = normalize_image(resized)
            inputs = torch.from_numpy(np.stack([image01, frangi_channel(image01)])[None]).to(device)
            with torch.inference_mode():
                probability = torch.sigmoid(model(inputs))[0, 0].cpu().numpy()
            probability = cv2.resize(probability, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            safe_name = "__".join(path.relative_to(root).with_suffix("").parts)
            Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255)).save(destination / f"{safe_name}_probability.png")
            Image.fromarray(np.uint8(probability >= 0.5) * 255).save(destination / f"{safe_name}_mask.png")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vessmap-root", type=Path, default=ROOT / "data" / "VessMAP")
    p.add_argument("--annotator", choices=["annotator1", "annotator2"], default="annotator1")
    p.add_argument("--external-roots", type=Path, nargs="*", default=[ROOT / "data" / "others", ROOT / "data" / "external"])
    p.add_argument("--output", type=Path, default=ROOT / "unet_runs" / "vessmap_frangi_2ch")
    p.add_argument("--initial-checkpoint", type=Path, default=None,
                   help="Load model weights and continue training in a new run")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--augment-repeats", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args(); seed_everything(args.seed)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(args.vessmap_root.resolve(), args.annotator)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.seed, args.val_fraction, args.test_fraction)
    splits = {"train": train_pairs, "validation": val_pairs, "test": test_pairs}
    save_split(output, splits)
    config = vars(args).copy()
    config.update({"input_channels": ["normalized_grayscale", "frangi_vesselness"], "output_channels": ["binary_vessel_mask"], "split_counts": {k: len(v) for k, v in splits.items()}, "note": "Frangi is input-only; it is not applied to masks or outputs."})
    config = {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else v for k, v in config.items()}
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    epochs = 1 if args.smoke_test else args.epochs
    repeats = 1 if args.smoke_test else args.augment_repeats
    workers = 0 if args.smoke_test else args.workers
    print("Preparing augmented two-channel samples (Frangi is computed once)...")
    train_dataset = VessMAPDataset(train_pairs, args.size, True, repeats, cache=True)
    val_dataset = VessMAPDataset(val_pairs, args.size, False, cache=True)
    test_dataset = VessMAPDataset(test_pairs, args.size, False, cache=True)
    # Cached tensors are already resident in RAM; worker processes would only
    # duplicate memory and add Windows spawn/serialization overhead.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(args.base_channels).to(device); loss_fn = DiceBCELoss()
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint.resolve(), map_location=device, weights_only=False)
        state = initial["model"] if isinstance(initial, dict) and "model" in initial else initial
        model.load_state_dict(state)
        print(f"Loaded initial weights: {args.initial_checkpoint.resolve()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0; stale = 0; history: list[dict[str, float]] = []
    best_path = output / "best_model.pt"
    for epoch in range(1, epochs + 1):
        model.train(); running = seen = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(inputs), targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += float(loss) * len(inputs); seen += len(inputs)
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row); print(json.dumps(row))
        if val["dice"] > best_dice:
            best_dice = val["dice"]; stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val, "config": config}, best_path)
        else:
            stale += 1
            if stale >= args.patience and not args.smoke_test:
                break
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    predicted = predict_external(model, [p.resolve() for p in args.external_roots], output, device, args.size)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test": test_metrics, "external_images": predicted}, indent=2))


if __name__ == "__main__":
    main()
