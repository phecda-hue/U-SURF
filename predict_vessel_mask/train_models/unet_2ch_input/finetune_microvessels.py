import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from predict import load_model, select_device
from trainunet import (
    BCEDiceLoss,
    ImageMaskDataset,
    Sample,
    save_checkpoint,
    train_one_epoch,
    write_history,
)


# ============================================================
# 기본 경로
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_RUN = (
    ROOT
    / "unet_runs"
    / "rsom_pa_unet"
)

OUTPUT_DIR = (
    ROOT
    / "unet_runs"
    / "rsom_pa_unet_micro_finetune"
)


# ============================================================
# Argument
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune U-Net using hard microvessel samples."
    )

    # 경로
    parser.add_argument(
        "--source-run",
        default=str(SOURCE_RUN),
        help="기존 U-Net 학습 결과 폴더",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Fine-tuning 결과 저장 폴더",
    )

    # --------------------------------------------------------
    # 미세혈관 판정
    # --------------------------------------------------------

    parser.add_argument(
        "--width-threshold",
        type=float,
        default=4.0,
        help=(
            "GT skeleton에서 미세혈관으로 간주할 "
            "대략적인 직경 threshold"
        ),
    )

    parser.add_argument(
        "--medium-width-threshold",
        type=float,
        default=8.0,
        help=(
            "Upper local-diameter threshold for medium vessels. "
            "Thin <= width-threshold, medium <= this value, thick > this value."
        ),
    )

    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=0.2,
        help=(
            "원본 영상에서 약한 미세혈관 신호를 "
            "분석하기 위한 intensity threshold"
        ),
    )

    parser.add_argument(
        "--signal-percentile",
        type=float,
        default=20.0,
        help="Per-image vessel-intensity percentile used as the hard-point minimum signal.",
    )

    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--thin-dilation", type=int, default=2)
    parser.add_argument("--thin-weight", type=float, default=3.0)

    # 분석용 threshold
    # 실제 hard score에는 직접 사용하지 않음
    parser.add_argument(
        "--miss-threshold",
        type=float,
        default=0.3,
        help=(
            "미세혈관 미탐 비율을 기록하기 위한 "
            "probability threshold"
        ),
    )

    # validation metric 계산용 threshold
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.1,
        help="Validation binary prediction threshold",
    )

    # --------------------------------------------------------
    # Hard sample sampling
    # --------------------------------------------------------

    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=0.25,
        help=(
            "전체 training sample 중 hard sample 후보 비율. "
            "기본값 0.25 = 상위 25%%"
        ),
    )

    parser.add_argument(
        "--hard-sampling-ratio",
        type=float,
        default=0.5,
        help=(
            "한 epoch의 sampling 확률 중 hard sample 비율. "
            "기본값 0.5 = hard/general 50:50"
        ),
    )

    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0이면 모든 training batch 사용",
    )

    return parser.parse_args()


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 데이터 경로 복구
# ============================================================

def repair_path(recorded_path):
    path = Path(recorded_path)

    if path.is_file():
        return path

    anchors = (
        "augmented_training_dataset_3x",
        "Photoacoustic vascular image dataset",
    )

    for anchor in anchors:
        if anchor in path.parts:

            repaired = (
                ROOT
                / "data"
                / Path(
                    *path.parts[
                        path.parts.index(anchor):
                    ]
                )
            )

            if repaired.is_file():
                return repaired

    raise FileNotFoundError(recorded_path)


# ============================================================
# Split 로드
# ============================================================

def load_samples(manifest_path):

    splits = {
        name: []
        for name in (
            "train",
            "val_pa",
            "val_rsom",
            "test_pa",
        )
    }

    skipped_train = 0

    with manifest_path.open(
        encoding="utf-8-sig"
    ) as file:

        for row in csv.DictReader(file):

            if row["split"] not in splits:
                continue

            try:
                image_path = repair_path(
                    row["image_path"]
                )

                mask_path = repair_path(
                    row["mask_path"]
                )

            except FileNotFoundError:

                if row["split"] == "train":
                    skipped_train += 1
                    continue

                raise

            splits[row["split"]].append(
                Sample(
                    file_name=row["file_name"],
                    image_path=image_path,
                    mask_path=mask_path,
                    source=row["source"],
                    source_id=row["source_id"],
                    variant=row["variant"],
                )
            )

    if skipped_train:
        print(
            f"Skipped missing training samples: "
            f"{skipped_train}"
        )

    for split_name, samples in splits.items():
        print(
            f"{split_name}: {len(samples)} samples"
        )

    return splits


# ============================================================
# Hard sample scoring
# ============================================================

@torch.inference_mode()
def score_training_samples(
    model,
    samples,
    device,
    batch_size,
    workers,
    settings,
):

    loader = DataLoader(
        ImageMaskDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()

    records = []
    patch_metadata = []

    offset = 0

    for images, masks in loader:

        images_device = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images_device)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()[:, 0]
        )

        images_np = (
            images
            .numpy()[:, 0]
        )

        masks_np = (
            masks
            .numpy()[:, 0]
            > 0.5
        )

        for index in range(len(images_np)):

            gt = masks_np[index]

            # --------------------------------------------
            # GT skeleton
            # --------------------------------------------

            skeleton = skeletonize(gt)

            # 각 혈관 위치에서 대략적인 반경
            distance = distance_transform_edt(gt)

            # 대략적인 직경
            diameter = 2.0 * distance

            # 얇은 혈관 skeleton
            thin_skeleton = (
                skeleton
                & (
                    diameter
                    <= settings.width_threshold
                )
            )

            medium_skeleton = (
                skeleton
                & (diameter > settings.width_threshold)
                & (diameter <= settings.medium_width_threshold)
            )

            thick_skeleton = (
                skeleton
                & (diameter > settings.medium_width_threshold)
            )

            thin_length = int(
                thin_skeleton.sum()
            )

            skeleton_length = int(
                skeleton.sum()
            )

            # --------------------------------------------
            # 미세혈관 양
            #
            # 이미지 전체에서 미세혈관 skeleton이
            # 차지하는 비율
            # --------------------------------------------

            micro_amount = (
                thin_length
                / max(gt.size, 1)
            )

            # 전체 혈관 skeleton 중
            # 미세혈관 비율
            thin_ratio = (
                thin_length
                / max(skeleton_length, 1)
            )

            # --------------------------------------------
            # 미세혈관 난이도 계산
            # --------------------------------------------

            if thin_length > 0:

                thin_probs = (
                    probabilities[index][
                        thin_skeleton
                    ]
                )

                thin_intensities = (
                    images_np[index][
                        thin_skeleton
                    ]
                )

                # 원본 영상에서
                # 약한 미세혈관의 비율
                weak_thin_ratio = float(
                    np.mean(
                        thin_intensities
                        < settings.intensity_threshold
                    )
                )

                # threshold 기반 미탐 비율
                # 분석 및 CSV 기록용
                missed_thin_ratio = float(
                    np.mean(
                        thin_probs
                        < settings.miss_threshold
                    )
                )

                # ----------------------------------------
                # Soft difficulty
                #
                # probability가 낮을수록 높은 점수
                #
                # p = 0.9 -> difficulty 0.1
                # p = 0.2 -> difficulty 0.8
                # ----------------------------------------

                missed_thin_score = float(
                    np.mean(
                        1.0 - thin_probs
                    )
                )

                mean_thin_probability = float(
                    np.mean(thin_probs)
                )

                vessel_intensities = images_np[index][gt]
                minimum_signal = float(
                    np.percentile(vessel_intensities, settings.signal_percentile)
                )
                hard_mask = (
                    thin_skeleton
                    & (probabilities[index] < settings.miss_threshold)
                    & (images_np[index] > minimum_signal)
                )
                normal_mask = (
                    skeleton
                    & (probabilities[index] >= settings.miss_threshold)
                )

            else:

                weak_thin_ratio = 0.0
                missed_thin_ratio = 0.0
                missed_thin_score = 0.0
                mean_thin_probability = 0.0
                minimum_signal = 0.0
                hard_mask = np.zeros_like(gt, dtype=bool)
                normal_mask = np.zeros_like(gt, dtype=bool)

            # --------------------------------------------
            # 최종 Hard score
            #
            # 미세혈관이 많고
            # 현재 모델이 낮은 확률을 출력할수록
            # 높은 점수
            # --------------------------------------------

            score = (
                micro_amount
                * missed_thin_score
            )

            sample_index = offset + index

            sample = samples[sample_index]

            records.append(
                {
                    "index": sample_index,

                    "file_name":
                        sample.file_name,

                    "source":
                        sample.source,

                    "source_id":
                        sample.source_id,

                    "variant":
                        sample.variant,

                    "thin_length":
                        thin_length,

                    "skeleton_length":
                        skeleton_length,

                    "thin_ratio":
                        thin_ratio,

                    "micro_amount":
                        micro_amount,

                    "weak_thin_ratio":
                        weak_thin_ratio,

                    "mean_thin_probability":
                        mean_thin_probability,

                    "missed_thin_ratio":
                        missed_thin_ratio,

                    "missed_thin_score":
                        missed_thin_score,

                    "score":
                        score,

                    "minimum_signal":
                        minimum_signal,

                    "hard_point_count":
                        int(hard_mask.sum()),
                }
            )

            patch_metadata.append(
                {
                    "hard_points": np.argwhere(hard_mask),
                    "normal_points": np.argwhere(normal_mask),
                    "medium_points": np.argwhere(medium_skeleton),
                    "thick_points": np.argwhere(thick_skeleton),
                    "thin_region": binary_dilation(
                        thin_skeleton,
                        iterations=settings.thin_dilation,
                    ),
                }
            )

        offset += len(images_np)

        print(
            f"Hard-sample scoring: "
            f"{offset}/{len(samples)}",
            flush=True,
        )

    return records, patch_metadata


class MicrovesselPatchDataset(Dataset):
    """Sample 50% thin, 30% medium, and 20% thick-vessel patches."""

    def __init__(self, samples, metadata, patch_size, thin_weight):
        self.base = ImageMaskDataset(samples)
        self.metadata = metadata
        self.patch_size = patch_size
        self.thin_weight = thin_weight
        self.hard_indices = [
            index for index, item in enumerate(metadata) if len(item["hard_points"])
        ]
        self.medium_indices = [
            index for index, item in enumerate(metadata) if len(item["medium_points"])
        ]
        self.thick_indices = [
            index for index, item in enumerate(metadata) if len(item["thick_points"])
        ]
        if not self.hard_indices:
            raise RuntimeError("No signal-qualified missed thin-vessel points were found.")
        if not self.medium_indices:
            raise RuntimeError("No medium-vessel skeleton points were found.")
        if not self.thick_indices:
            raise RuntimeError("No thick-vessel skeleton points were found.")

    def __len__(self):
        return len(self.base)

    def _crop(self, array, center_y, center_x):
        half = self.patch_size // 2
        y0, x0 = center_y - half, center_x - half
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size
        pad_top, pad_left = max(0, -y0), max(0, -x0)
        pad_bottom = max(0, y1 - array.shape[-2])
        pad_right = max(0, x1 - array.shape[-1])
        if pad_top or pad_bottom or pad_left or pad_right:
            array = F.pad(array, (pad_left, pad_right, pad_top, pad_bottom))
            y0, y1 = y0 + pad_top, y1 + pad_top
            x0, x1 = x0 + pad_left, x1 + pad_left
        return array[..., y0:y1, x0:x1]

    def __getitem__(self, index):
        # Every group of ten samples contains exactly 5 thin, 3 medium,
        # and 2 thick-vessel centered patches.
        category = index % 10
        if category < 5:
            sample_index = random.choice(self.hard_indices)
            points = self.metadata[sample_index]["hard_points"]
            center_y, center_x = points[np.random.randint(len(points))]
        elif category < 8:
            sample_index = random.choice(self.medium_indices)
            points = self.metadata[sample_index]["medium_points"]
            center_y, center_x = points[np.random.randint(len(points))]
        else:
            sample_index = random.choice(self.thick_indices)
            points = self.metadata[sample_index]["thick_points"]
            center_y, center_x = points[np.random.randint(len(points))]

        image, mask = self.base[sample_index]
        thin_region = torch.from_numpy(
            self.metadata[sample_index]["thin_region"].astype(np.float32)
        ).unsqueeze(0)
        image_patch = self._crop(image, int(center_y), int(center_x))
        mask_patch = self._crop(mask, int(center_y), int(center_x))
        thin_patch = self._crop(thin_region, int(center_y), int(center_x))
        pixel_weight = 1.0 + (self.thin_weight - 1.0) * thin_patch
        return image_patch, mask_patch, pixel_weight


class ThinWeightedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets, pixel_weights=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if pixel_weights is not None:
            bce = bce * pixel_weights
        bce = bce.mean()
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


def train_patch_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, max_batches):
    model.train()
    loss_sum = sample_count = tp = fp = fn = 0.0
    for batch_index, (images, masks, weights) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks, weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        prediction = torch.sigmoid(logits.detach()) >= 0.5
        truth = masks >= 0.5
        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        tp += (prediction & truth).sum().item()
        fp += (prediction & ~truth).sum().item()
        fn += (~prediction & truth).sum().item()
    epsilon = 1e-8
    return {
        "loss": loss_sum / max(sample_count, 1),
        "dice": (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon),
        "iou": (tp + epsilon) / (tp + fp + fn + epsilon),
        "precision": (tp + epsilon) / (tp + fp + epsilon),
        "recall": (tp + epsilon) / (tp + fn + epsilon),
    }


# ============================================================
# Hard sample 선택
# ============================================================

def select_hard_samples(
    records,
    hard_fraction,
):

    # 미세혈관이 실제로 존재하고
    # score > 0인 샘플만 후보
    eligible = [
        row
        for row in records
        if (
            row["thin_length"] > 0
            and row["score"] > 0
        )
    ]

    if not eligible:
        raise RuntimeError(
            "No eligible hard microvessel samples found."
        )

    ranked = sorted(
        eligible,
        key=lambda row: row["score"],
        reverse=True,
    )

    requested_count = max(
        1,
        round(
            len(records)
            * hard_fraction
        ),
    )

    hard_count = min(
        len(ranked),
        requested_count,
    )

    hard_indices = {
        row["index"]
        for row in ranked[:hard_count]
    }

    for row in records:
        row["is_hard"] = int(
            row["index"]
            in hard_indices
        )

    return hard_indices


# ============================================================
# Hard sample score CSV 저장
# ============================================================

def write_scores(
    path,
    records,
):

    if not records:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            sorted(
                records,
                key=lambda row: row["score"],
                reverse=True,
            )
        )


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    samples,
    batch_size,
    workers,
    device,
    sampler=None,
):

    return DataLoader(
        ImageMaskDataset(samples),
        batch_size=batch_size,

        # sampler 사용 시 shuffle 불가
        shuffle=False,

        sampler=sampler,

        num_workers=workers,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )


# ============================================================
# Hard/general mixture sampler
# ============================================================

def hard_mixture_sampler(
    sample_count,
    hard_indices,
    hard_ratio,
    seed,
):

    hard_count = len(
        hard_indices
    )

    general_count = (
        sample_count
        - hard_count
    )

    if hard_count == 0:
        raise ValueError(
            "No hard samples were selected."
        )

    # 모든 sample이 hard sample인 경우
    if general_count == 0:

        weights = np.ones(
            sample_count,
            dtype=np.float64,
        )

    else:

        weights = np.array(
            [
                (
                    hard_ratio
                    / hard_count
                )
                if index in hard_indices

                else
                (
                    (1.0 - hard_ratio)
                    / general_count
                )

                for index
                in range(sample_count)
            ],
            dtype=np.float64,
        )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    return WeightedRandomSampler(
        weights=weights,

        # 한 epoch당 원래 dataset 크기와
        # 동일한 수의 sample 사용
        num_samples=sample_count,

        replacement=True,

        generator=generator,
    )


# ============================================================
# Metric utility
# ============================================================

def safe_divide(
    numerator,
    denominator,
    default=0.0,
):

    if denominator == 0:
        return default

    return (
        numerator
        / denominator
    )


def nanmean_or_zero(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return 0.0

    valid = values[
        ~np.isnan(values)
    ]

    if len(valid) == 0:
        return 0.0

    return float(
        valid.mean()
    )


# ============================================================
# clDice
# ============================================================

def calculate_cldice(
    prediction,
    ground_truth,
):

    prediction = (
        prediction.astype(bool)
    )

    ground_truth = (
        ground_truth.astype(bool)
    )

    # 둘 다 비어있는 경우
    if (
        prediction.sum() == 0
        and ground_truth.sum() == 0
    ):
        return 1.0

    pred_skeleton = skeletonize(
        prediction
    )

    gt_skeleton = skeletonize(
        ground_truth
    )

    # --------------------------------------------
    # Topology precision
    #
    # 예측 skeleton 중
    # GT 내부에 있는 비율
    # --------------------------------------------

    if pred_skeleton.sum() > 0:

        topology_precision = (
            (
                pred_skeleton
                & ground_truth
            ).sum()
            / pred_skeleton.sum()
        )

    else:

        topology_precision = 0.0

    # --------------------------------------------
    # Topology sensitivity
    #
    # GT skeleton 중
    # prediction 내부에 있는 비율
    # --------------------------------------------

    if gt_skeleton.sum() > 0:

        topology_sensitivity = (
            (
                gt_skeleton
                & prediction
            ).sum()
            / gt_skeleton.sum()
        )

    else:

        topology_sensitivity = 1.0

    denominator = (
        topology_precision
        + topology_sensitivity
    )

    if denominator == 0:
        return 0.0

    cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / denominator
    )

    return float(cldice)


# ============================================================
# Fine-tuning validation
# ============================================================

@torch.inference_mode()
def evaluate_finetune(
    model,
    loader,
    criterion,
    device,
    use_amp,
    metric_threshold,
    width_threshold,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Pixel-level global counts
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Structure-level metrics
    cldice_values = []
    skeleton_recall_values = []
    thin_recall_values = []

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        batch_size = images.size(0)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities
            >= metric_threshold
        )

        ground_truth = (
            masks
            >= 0.5
        )

        # --------------------------------------------
        # Loss
        # --------------------------------------------

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_samples += batch_size

        # --------------------------------------------
        # Pixel-level metrics
        # --------------------------------------------

        total_tp += int(
            (
                predictions
                & ground_truth
            )
            .sum()
            .item()
        )

        total_fp += int(
            (
                predictions
                & ~ground_truth
            )
            .sum()
            .item()
        )

        total_fn += int(
            (
                ~predictions
                & ground_truth
            )
            .sum()
            .item()
        )

        # CPU numpy
        predictions_np = (
            predictions
            .cpu()
            .numpy()[:, 0]
        )

        gt_np = (
            ground_truth
            .cpu()
            .numpy()[:, 0]
        )

        # --------------------------------------------
        # 각 이미지별 구조 metric
        # --------------------------------------------

        for prediction, gt in zip(
            predictions_np,
            gt_np,
        ):

            prediction = (
                prediction.astype(bool)
            )

            gt = (
                gt.astype(bool)
            )

            # ------------------------
            # clDice
            # ------------------------

            cldice = calculate_cldice(
                prediction,
                gt,
            )

            cldice_values.append(
                cldice
            )

            # ------------------------
            # Skeleton Recall
            # ------------------------

            gt_skeleton = skeletonize(
                gt
            )

            if gt_skeleton.sum() > 0:

                skeleton_recall = (
                    (
                        gt_skeleton
                        & prediction
                    ).sum()
                    / gt_skeleton.sum()
                )

                skeleton_recall_values.append(
                    float(
                        skeleton_recall
                    )
                )

            # ------------------------
            # Thin-vessel Recall
            # ------------------------

            distance = (
                distance_transform_edt(
                    gt
                )
            )

            diameter = (
                2.0
                * distance
            )

            thin_skeleton = (
                gt_skeleton
                & (
                    diameter
                    <= width_threshold
                )
            )

            if thin_skeleton.sum() > 0:

                thin_recall = (
                    (
                        thin_skeleton
                        & prediction
                    ).sum()
                    / thin_skeleton.sum()
                )

                thin_recall_values.append(
                    float(
                        thin_recall
                    )
                )

    # ========================================================
    # 최종 metric
    # ========================================================

    precision = safe_divide(
        total_tp,
        total_tp + total_fp,
    )

    recall = safe_divide(
        total_tp,
        total_tp + total_fn,
    )

    dice = safe_divide(
        2 * total_tp,
        (
            2 * total_tp
            + total_fp
            + total_fn
        ),
    )

    iou = safe_divide(
        total_tp,
        (
            total_tp
            + total_fp
            + total_fn
        ),
    )

    return {
        "loss":
            total_loss
            / max(
                total_samples,
                1,
            ),

        "dice":
            float(dice),

        "iou":
            float(iou),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "cldice":
            nanmean_or_zero(
                cldice_values
            ),

        "skeleton_recall":
            nanmean_or_zero(
                skeleton_recall_values
            ),

        "thin_recall":
            nanmean_or_zero(
                thin_recall_values
            ),
    }


# ============================================================
# Metric 출력
# ============================================================

def format_finetune_metrics(
    metrics,
):

    return (
        f"Loss={metrics['loss']:.4f}, "
        f"Dice={metrics['dice']:.4f}, "
        f"IoU={metrics['iou']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"clDice={metrics['cldice']:.4f}, "
        f"SkeletonRecall="
        f"{metrics['skeleton_recall']:.4f}, "
        f"ThinRecall="
        f"{metrics['thin_recall']:.4f}"
    )


# ============================================================
# PA + RSOM validation 평균
# ============================================================

def mean_validation_metrics(
    pa_metrics,
    rsom_metrics,
):

    keys = (
        "loss",
        "dice",
        "iou",
        "precision",
        "recall",
        "cldice",
        "skeleton_recall",
        "thin_recall",
    )

    return {
        key: (
            pa_metrics[key]
            + rsom_metrics[key]
        ) / 2.0

        for key in keys
    }


# ============================================================
# Best model 선택용 종합 score
# ============================================================

def calculate_validation_score(
    metrics,
):
    """
    구조 보존을 중요하게 평가합니다.

    Dice              : 25%
    clDice            : 25%
    Skeleton Recall   : 20%
    Thin-vessel Recall: 20%
    Precision         : 10%

    Recall만 지나치게 높이고
    오탐이 증가하는 것을 방지하기 위해
    Precision도 일부 포함합니다.
    """

    score = (
        0.25
        * metrics["dice"]

        + 0.25
        * metrics["cldice"]

        + 0.20
        * metrics["skeleton_recall"]

        + 0.20
        * metrics["thin_recall"]

        + 0.10
        * metrics["precision"]
    )

    return float(score)


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Argument 검사
    # --------------------------------------------------------

    if not (
        0
        < args.hard_fraction
        < 1
    ):
        raise ValueError(
            "--hard-fraction must be between 0 and 1."
        )

    if not (
        0
        < args.hard_sampling_ratio
        < 1
    ):
        raise ValueError(
            "--hard-sampling-ratio must be between 0 and 1."
        )

    if args.width_threshold <= 0:
        raise ValueError("--width-threshold must be greater than 0.")

    if args.medium_width_threshold <= args.width_threshold:
        raise ValueError(
            "--medium-width-threshold must be greater than --width-threshold."
        )

    set_seed(
        args.seed
    )

    source_run = Path(
        args.source_run
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # config 저장
    with (
        output_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Dataset split
    # ========================================================

    splits = load_samples(
        source_run
        / "splits.csv"
    )

    # ========================================================
    # Device
    # ========================================================

    device = select_device(
        args.device
    )

    use_amp = (
        device.type
        == "cuda"
    )

    # ========================================================
    # 기존 best model 로드
    # ========================================================

    source_model_path = (
        source_run
        / "best_model.pt"
    )

    model, _ = load_model(
        source_model_path,
        device,
    )

    print(
        f"\nLoaded model:"
        f"\n  {source_model_path}"
        f"\nDevice: {device}"
    )

    # ========================================================
    # Training sample hard score 계산
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "Scoring training samples..."
    )

    print(
        "========================================"
    )

    records, patch_metadata = score_training_samples(
        model=model,

        samples=splits["train"],

        device=device,

        batch_size=args.batch_size,

        workers=args.workers,

        settings=args,
    )

    hard_indices = select_hard_samples(
        records,
        args.hard_fraction,
    )

    write_scores(
        output_dir
        / "microvessel_scores.csv",

        records,
    )

    hard_records = [
        row
        for row in records
        if row["is_hard"]
    ]

    hard_scores = [
        row["score"]
        for row in hard_records
    ]

    print(
        f"\nHard samples:"
        f" {len(hard_indices)}"
        f"/{len(records)}"
    )

    print(
        f"Hard score range:"
        f" {min(hard_scores):.6g}"
        f" - "
        f"{max(hard_scores):.6g}"
    )

    print(
        "\nTop 10 hard samples:"
    )

    for row in sorted(
        hard_records,
        key=lambda x: x["score"],
        reverse=True,
    )[:10]:

        print(
            f"  "
            f"{row['file_name']} | "
            f"score={row['score']:.6f} | "
            f"thin_length="
            f"{row['thin_length']} | "
            f"thin_prob="
            f"{row['mean_thin_probability']:.4f}"
        )

    # ========================================================
    # DataLoader
    # ========================================================

    patch_dataset = MicrovesselPatchDataset(
        splits["train"],
        patch_metadata,
        args.patch_size,
        args.thin_weight,
    )

    print(
        "\nPatch sampling ratio: "
        "thin 50% / medium 30% / thick 20%"
    )
    print(
        "Vessel diameter ranges: "
        f"thin <= {args.width_threshold:g}px, "
        f"medium <= {args.medium_width_threshold:g}px, "
        f"thick > {args.medium_width_threshold:g}px"
    )
    print(
        "Candidate images: "
        f"thin={len(patch_dataset.hard_indices)}, "
        f"medium={len(patch_dataset.medium_indices)}, "
        f"thick={len(patch_dataset.thick_indices)}"
    )

    train_loader = DataLoader(
        patch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    pa_val_loader = make_loader(
        splits["val_pa"],
        args.batch_size,
        args.workers,
        device,
    )

    rsom_val_loader = make_loader(
        splits["val_rsom"],
        args.batch_size,
        args.workers,
        device,
    )

    test_loader = make_loader(
        splits["test_pa"],
        args.batch_size,
        args.workers,
        device,
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = ThinWeightedBCEDiceLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=args.learning_rate,

        weight_decay=(
            args.weight_decay
        ),
    )

    # ========================================================
    # Scheduler
    #
    # validation composite score가
    # 개선되지 않으면 LR 감소
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # ========================================================
    # AMP scaler
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # ========================================================
    # Baseline 평가
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "Baseline evaluation"
    )

    print(
        "========================================"
    )

    baseline_pa = evaluate_finetune(
        model=model,

        loader=pa_val_loader,

        criterion=criterion,

        device=device,

        use_amp=use_amp,

        metric_threshold=(
            args.metric_threshold
        ),

        width_threshold=(
            args.width_threshold
        ),
    )

    baseline_rsom = evaluate_finetune(
        model=model,

        loader=rsom_val_loader,

        criterion=criterion,

        device=device,

        use_amp=use_amp,

        metric_threshold=(
            args.metric_threshold
        ),

        width_threshold=(
            args.width_threshold
        ),
    )

    baseline_mean = (
        mean_validation_metrics(
            baseline_pa,
            baseline_rsom,
        )
    )

    baseline_score = (
        calculate_validation_score(
            baseline_mean
        )
    )

    print(
        "\nBaseline PA val:"
    )

    print(
        "  "
        + format_finetune_metrics(
            baseline_pa
        )
    )

    print(
        "\nBaseline RSOM val:"
    )

    print(
        "  "
        + format_finetune_metrics(
            baseline_rsom
        )
    )

    print(
        f"\nBaseline composite score:"
        f" {baseline_score:.6f}"
    )

    # ========================================================
    # Baseline checkpoint 저장
    # ========================================================

    baseline_row = {
        "epoch": 0,

        "learning_rate":
            args.learning_rate,

        **{
            f"pa_val_{key}":
                value

            for key, value
            in baseline_pa.items()
        },

        **{
            f"rsom_val_{key}":
                value

            for key, value
            in baseline_rsom.items()
        },

        **{
            f"mean_val_{key}":
                value

            for key, value
            in baseline_mean.items()
        },

        "validation_score":
            baseline_score,
    }

    checkpoints_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_epoch_path = (
        checkpoints_dir
        / "model_epoch_0000.pt"
    )

    save_checkpoint(
        baseline_epoch_path,

        model,

        0,

        baseline_row,
    )

    save_checkpoint(
        output_dir
        / "baseline_model.pt",

        model,

        0,

        baseline_row,
    )

    print(
        "Saved checkpoint "
        f"for epoch 0: {baseline_epoch_path}"
    )

    # fine-tuning이 개선되지 않아도
    # best_model.pt가 존재하도록 baseline 저장
    save_checkpoint(
        output_dir
        / "best_model.pt",

        model,

        0,

        baseline_row,
    )

    # ========================================================
    # Fine-tuning
    # ========================================================

    best_score = baseline_score

    best_epoch = 0

    no_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"Fine-tune epoch "
            f"{epoch}/{args.epochs}"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = (
            train_patch_epoch(
                model,

                train_loader,

                optimizer,

                criterion,

                scaler,

                device,

                use_amp,

                args.max_train_batches,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        pa_metrics = evaluate_finetune(
            model=model,

            loader=pa_val_loader,

            criterion=criterion,

            device=device,

            use_amp=use_amp,

            metric_threshold=(
                args.metric_threshold
            ),

            width_threshold=(
                args.width_threshold
            ),
        )

        rsom_metrics = evaluate_finetune(
            model=model,

            loader=rsom_val_loader,

            criterion=criterion,

            device=device,

            use_amp=use_amp,

            metric_threshold=(
                args.metric_threshold
            ),

            width_threshold=(
                args.width_threshold
            ),
        )

        mean_metrics = (
            mean_validation_metrics(
                pa_metrics,
                rsom_metrics,
            )
        )

        validation_score = (
            calculate_validation_score(
                mean_metrics
            )
        )

        # Scheduler
        scheduler.step(
            validation_score
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        row = {
            "epoch":
                epoch,

            "learning_rate":
                optimizer
                .param_groups[0]["lr"],

            **{
                f"train_{key}":
                    value

                for key, value
                in train_metrics.items()
            },

            **{
                f"pa_val_{key}":
                    value

                for key, value
                in pa_metrics.items()
            },

            **{
                f"rsom_val_{key}":
                    value

                for key, value
                in rsom_metrics.items()
            },

            **{
                f"mean_val_{key}":
                    value

                for key, value
                in mean_metrics.items()
            },

            "validation_score":
                validation_score,
        }

        history.append(
            row
        )

        write_history(
            output_dir
            / "history.csv",

            history,
        )

        epoch_checkpoint_path = (
            checkpoints_dir
            / f"model_epoch_{epoch:04d}.pt"
        )

        # Keep a permanent checkpoint for every completed epoch.
        save_checkpoint(
            epoch_checkpoint_path,

            model,

            epoch,

            row,
        )

        # 항상 마지막 checkpoint 저장
        save_checkpoint(
            output_dir
            / "last_model.pt",

            model,

            epoch,

            row,
        )

        print(
            "\nSaved checkpoint "
            f"for epoch {epoch}: "
            f"{epoch_checkpoint_path}"
        )

        # ----------------------------------------------------
        # 출력
        # ----------------------------------------------------

        print(
            "\nTrain:"
        )

        print(
            "  "
            + str(
                train_metrics
            )
        )

        print(
            "\nPA val:"
        )

        print(
            "  "
            + format_finetune_metrics(
                pa_metrics
            )
        )

        print(
            "\nRSOM val:"
        )

        print(
            "  "
            + format_finetune_metrics(
                rsom_metrics
            )
        )

        print(
            "\nMean validation:"
        )

        print(
            "  "
            + format_finetune_metrics(
                mean_metrics
            )
        )

        print(
            f"\nComposite score:"
            f" {validation_score:.6f}"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            validation_score
            > best_score
        ):

            best_score = (
                validation_score
            )

            best_epoch = epoch

            no_improvement = 0

            save_checkpoint(
                output_dir
                / "best_model.pt",

                model,

                epoch,

                row,
            )

            print(
                "\nSaved new best model"
            )

            print(
                f"  Epoch:"
                f" {epoch}"
            )

            print(
                "  Permanent checkpoint: "
                f"{epoch_checkpoint_path}"
            )

            print(
                f"  Score:"
                f" {best_score:.6f}"
            )

        else:

            no_improvement += 1

            print(
                f"\nNo improvement:"
                f" {no_improvement}"
                f"/{args.early_stopping}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            no_improvement
            >= args.early_stopping
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # Best model test
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "Final test evaluation"
    )

    print(
        "========================================"
    )

    best_path = (
        output_dir
        / "best_model.pt"
    )

    best_model, _ = load_model(
        best_path,
        device,
    )

    test_metrics = evaluate_finetune(
        model=best_model,

        loader=test_loader,

        criterion=criterion,

        device=device,

        use_amp=use_amp,

        metric_threshold=(
            args.metric_threshold
        ),

        width_threshold=(
            args.width_threshold
        ),
    )

    print(
        "\nBest model:"
    )

    print(
        f"  Epoch:"
        f" {best_epoch}"
    )

    print(
        f"  Validation score:"
        f" {best_score:.6f}"
    )

    print(
        "\nFine-tuned PA test:"
    )

    print(
        "  "
        + format_finetune_metrics(
            test_metrics
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {output_dir}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
