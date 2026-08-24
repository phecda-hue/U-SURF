import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from predict import (
    create_overlay,
    load_model,
    normalize_image,
    predict as predict_probability,
    read_grayscale,
    select_device,
)
from trainunet import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PA_ROOT,
    Sample,
    load_augmented_splits,
    load_pa_test_samples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "unet_runs" / "rsom_pa_unet"
DEFAULT_MODEL = DEFAULT_RUN_DIR / "best_model.pt"
DEFAULT_SPLITS_CSV = DEFAULT_RUN_DIR / "splits.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "evaluation"
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class EvalSample:
    split: str
    file_name: str
    source: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net vessel segmentation model."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to a trained U-Net checkpoint.",
    )
    parser.add_argument(
        "--splits-csv",
        default=str(DEFAULT_SPLITS_CSV),
        help=(
            "splits.csv created by trainunet.py. Ignored when "
            "--image-dir and --mask-dir are provided."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        help=(
            "Split to evaluate. Can be used repeatedly or as a comma-separated "
            "list. Default: val_pa,val_rsom,test_pa. Use 'all' for every split "
            "in splits.csv."
        ),
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also evaluate the training split.",
    )
    parser.add_argument(
        "--image-dir",
        help="Optional custom image directory for folder-based evaluation.",
    )
    parser.add_argument(
        "--mask-dir",
        help="Optional custom mask directory for folder-based evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where evaluation CSV/JSON files are saved.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        help='Device such as "auto", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate only the first N samples. 0 means all samples.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted masks, probability maps, and overlays.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N images.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_existing_dataset_path(value: str | Path) -> Path:
    path = resolve_path(value)
    if path.exists():
        return path

    try:
        relative_to_root = path.relative_to(ROOT)
    except ValueError:
        relative_to_root = None

    if relative_to_root is not None:
        data_path = ROOT / "data" / relative_to_root
        if data_path.exists():
            return data_path

    return path


def preferred_data_root() -> Path:
    data_root = ROOT / "data" / DEFAULT_DATA_ROOT.name
    return data_root if data_root.exists() else DEFAULT_DATA_ROOT


def preferred_pa_root() -> Path:
    data_pa_root = ROOT / "data" / DEFAULT_PA_ROOT.relative_to(ROOT)
    return data_pa_root if data_pa_root.exists() else DEFAULT_PA_ROOT


def parse_requested_splits(values: list[str] | None) -> list[str]:
    if not values:
        return ["val_pa", "val_rsom", "test_pa"]

    splits: list[str] = []
    for value in values:
        for split in value.split(","):
            split = split.strip()
            if split:
                splits.append(split)
    if "all" in splits:
        return ["all"]
    return splits


def sample_to_eval(split: str, sample: Sample) -> EvalSample:
    return EvalSample(
        split=split,
        file_name=sample.file_name,
        source=sample.source,
        image_path=sample.image_path,
        mask_path=sample.mask_path,
    )


def load_default_samples_without_csv(requested_splits: list[str]) -> list[EvalSample]:
    train_samples, pa_val_samples, rsom_val_samples = load_augmented_splits(
        data_root=preferred_data_root(),
        pa_val_group="patch_7",
        rsom_val_start=240,
        rsom_val_end=279,
        rsom_val_buffer=10,
    )
    test_samples = load_pa_test_samples(preferred_pa_root())
    split_map = {
        "train": [sample_to_eval("train", sample) for sample in train_samples],
        "val_pa": [sample_to_eval("val_pa", sample) for sample in pa_val_samples],
        "val_rsom": [
            sample_to_eval("val_rsom", sample) for sample in rsom_val_samples
        ],
        "test_pa": [sample_to_eval("test_pa", sample) for sample in test_samples],
    }

    if requested_splits == ["all"]:
        requested_splits = list(split_map)

    samples: list[EvalSample] = []
    for split in requested_splits:
        if split not in split_map:
            raise ValueError(
                f"Unknown split '{split}'. Available: {', '.join(split_map)}"
            )
        samples.extend(split_map[split])
    return samples


def load_samples_from_splits_csv(
    splits_csv: Path, requested_splits: list[str]
) -> list[EvalSample]:
    rows: list[dict[str, str]] = []
    with splits_csv.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    available_splits = sorted({row["split"] for row in rows})
    selected_splits = (
        available_splits if requested_splits == ["all"] else requested_splits
    )

    samples: list[EvalSample] = []
    for row in rows:
        if row["split"] not in selected_splits:
            continue
        image_path = resolve_existing_dataset_path(row["image_path"])
        mask_path = resolve_existing_dataset_path(row["mask_path"])
        samples.append(
            EvalSample(
                split=row["split"],
                file_name=row["file_name"],
                source=row.get("source", ""),
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    missing = [split for split in selected_splits if split not in available_splits]
    if missing:
        raise ValueError(
            f"Missing split(s) in {splits_csv}: {', '.join(missing)}\n"
            f"Available: {', '.join(available_splits)}"
        )
    return samples


def find_matching_mask(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    relative_path = image_path.relative_to(image_root)
    exact_path = mask_root / relative_path
    if exact_path.is_file():
        return exact_path

    candidates = [
        path
        for path in mask_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem == image_path.stem
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Mask not found for image: {image_path}")
    raise ValueError(
        f"Multiple masks with the same stem were found for {image_path.name}: "
        + ", ".join(str(path) for path in candidates[:5])
    )


def load_samples_from_folders(image_dir: Path, mask_dir: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = find_matching_mask(image_path, image_dir, mask_dir)
        samples.append(
            EvalSample(
                split="custom",
                file_name=image_path.relative_to(image_dir).as_posix(),
                source="custom",
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return samples


def load_samples(args: argparse.Namespace) -> list[EvalSample]:
    requested_splits = parse_requested_splits(args.split)
    if (
        args.include_train
        and requested_splits != ["all"]
        and "train" not in requested_splits
    ):
        requested_splits.append("train")

    if args.image_dir or args.mask_dir:
        if not args.image_dir or not args.mask_dir:
            raise ValueError("--image-dir and --mask-dir must be used together.")
        return load_samples_from_folders(
            resolve_path(args.image_dir), resolve_path(args.mask_dir)
        )

    splits_csv = resolve_path(args.splits_csv)
    if splits_csv.is_file():
        return load_samples_from_splits_csv(splits_csv, requested_splits)

    return load_default_samples_without_csv(requested_splits)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def count_confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = prediction.astype(bool)
    actual = truth.astype(bool)
    return {
        "tp": float(np.count_nonzero(predicted & actual)),
        "fp": float(np.count_nonzero(predicted & ~actual)),
        "fn": float(np.count_nonzero(~predicted & actual)),
        "tn": float(np.count_nonzero(~predicted & ~actual)),
    }


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "dice": dice,
        "iou": safe_divide(tp, tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "gt_foreground_fraction": safe_divide(tp + fn, tp + fp + fn + tn),
        "pred_foreground_fraction": safe_divide(tp + fp, tp + fp + fn + tn),
    }


def sanitize_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value).strip("_")


def save_prediction_outputs(
    output_dir: Path,
    sample: EvalSample,
    probability: np.ndarray,
    prediction: np.ndarray,
    image: np.ndarray,
) -> dict[str, str]:
    prediction_dir = output_dir / "predictions" / sample.split
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_name(Path(sample.file_name).with_suffix("").as_posix())

    mask_path = prediction_dir / f"{stem}_mask.png"
    probability_png_path = prediction_dir / f"{stem}_probability.png"
    probability_npy_path = prediction_dir / f"{stem}_probability.npy"
    overlay_path = prediction_dir / f"{stem}_overlay.png"

    mask_uint8 = prediction.astype(np.uint8) * 255
    probability_uint16 = np.clip(
        np.rint(probability * 65535.0), 0, 65535
    ).astype(np.uint16)

    Image.fromarray(mask_uint8).save(mask_path)
    Image.fromarray(probability_uint16).save(probability_png_path)
    np.save(probability_npy_path, probability.astype(np.float32))

    overlay = create_overlay(image, probability, prediction)
    overlay_uint8 = np.clip(np.rint(overlay * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8).save(overlay_path)

    return {
        "mask_output": str(mask_path),
        "probability_png_output": str(probability_png_path),
        "probability_npy_output": str(probability_npy_path),
        "overlay_output": str(overlay_path),
    }


def evaluate_sample(
    model: torch.nn.Module,
    sample: EvalSample,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, float | str]:
    raw_image = read_grayscale(sample.image_path)
    raw_mask = read_grayscale(sample.mask_path)
    image = normalize_image(raw_image)
    truth = raw_mask > 0

    probability = predict_probability(model, image, device)
    prediction = probability >= threshold

    if prediction.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch for {sample.file_name}: "
            f"prediction {prediction.shape} != mask {truth.shape}"
        )

    counts = count_confusion(prediction, truth)
    metrics = metrics_from_counts(counts)
    row: dict[str, float | str] = {
        "split": sample.split,
        "file_name": sample.file_name,
        "source": sample.source,
        "image_path": str(sample.image_path),
        "mask_path": str(sample.mask_path),
        "height": float(truth.shape[0]),
        "width": float(truth.shape[1]),
        **counts,
        **metrics,
    }

    if save_predictions:
        row.update(
            save_prediction_outputs(
                output_dir=output_dir,
                sample=sample,
                probability=probability,
                prediction=prediction,
                image=image,
            )
        )

    return row


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    numeric_keys = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "gt_foreground_fraction",
        "pred_foreground_fraction",
    ]
    count_keys = ["tp", "fp", "fn", "tn"]
    split_names = sorted({str(row["split"]) for row in rows})
    summary: dict[str, dict[str, float]] = {}

    for split in [*split_names, "overall"]:
        selected = rows if split == "overall" else [
            row for row in rows if row["split"] == split
        ]
        if not selected:
            continue

        pooled_counts = {
            key: float(sum(float(row[key]) for row in selected))
            for key in count_keys
        }
        pooled_metrics = metrics_from_counts(pooled_counts)
        split_summary: dict[str, float] = {
            "n_images": float(len(selected)),
            **{f"pooled_{key}": value for key, value in pooled_counts.items()},
            **{
                f"pooled_{key}": value
                for key, value in pooled_metrics.items()
            },
        }

        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            split_summary[f"mean_{key}"] = float(values.mean())
            split_summary[f"std_{key}"] = float(values.std(ddof=0))
            split_summary[f"min_{key}"] = float(values.min())
            split_summary[f"max_{key}"] = float(values.max())

        summary[split] = split_summary

    return summary


def write_per_image_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows to save.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, dict[str, float]]) -> None:
    fieldnames = ["split"]
    for metrics in summary.values():
        for key in metrics:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, metrics in summary.items():
            writer.writerow({"split": split, **metrics})


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    for split, metrics in summary.items():
        print(
            f"{split}: "
            f"n={metrics['n_images']:.0f} "
            f"pooled_dice={metrics['pooled_dice']:.4f} "
            f"mean_dice={metrics['mean_dice']:.4f} "
            f"pooled_iou={metrics['pooled_iou']:.4f} "
            f"mean_iou={metrics['mean_iou']:.4f} "
            f"precision={metrics['pooled_precision']:.4f} "
            f"recall={metrics['pooled_recall']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No evaluation samples were found.")

    for sample in samples:
        if not sample.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {sample.image_path}")
        if not sample.mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {sample.mask_path}")

    device = select_device(args.device)
    model, metadata = load_model(model_path, device)

    print(f"Device: {device}")
    print(
        f"Model: {model_path} "
        f"(base_channels={metadata['base_channels']}, "
        f"epoch={metadata.get('epoch')})"
    )
    print(f"Samples: {len(samples)}")
    print(f"Threshold: {args.threshold}")

    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            rows.append(
                evaluate_sample(
                    model=model,
                    sample=sample,
                    device=device,
                    threshold=args.threshold,
                    output_dir=output_dir,
                    save_predictions=args.save_predictions,
                )
            )
            if args.progress_every > 0 and (
                index == 1 or index % args.progress_every == 0 or index == len(samples)
            ):
                print(f"Evaluated {index}/{len(samples)}")

    summary = summarize_rows(rows)
    per_image_path = output_dir / "per_image_metrics.csv"
    summary_json_path = output_dir / "summary_metrics.json"
    summary_csv_path = output_dir / "summary_metrics.csv"

    write_per_image_csv(per_image_path, rows)
    write_summary_csv(summary_csv_path, summary)
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"Per-image metrics: {per_image_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
