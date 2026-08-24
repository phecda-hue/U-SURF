import argparse
import csv
import json
from pathlib import Path

import torch

from finetune_microvessels import (
    OUTPUT_DIR,
    SOURCE_RUN,
    ThinWeightedBCEDiceLoss,
    evaluate_finetune,
    format_finetune_metrics,
    load_samples,
    make_loader,
)
from predict import load_model, select_device


METRIC_KEYS = (
    "loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "cldice",
    "skeleton_recall",
    "thin_recall",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original model and every fine-tuning epoch "
            "checkpoint on exactly the same dataset split."
        )
    )
    parser.add_argument(
        "--source-run",
        default=str(SOURCE_RUN),
        help="Run directory containing the original best_model.pt and splits.csv.",
    )
    parser.add_argument(
        "--finetune-dir",
        default=str(OUTPUT_DIR),
        help="Fine-tuning output directory containing checkpoints/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val_pa", "val_rsom", "test_pa"),
        default=("test_pa",),
        help="Dataset split(s) used identically for every model.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric-threshold", type=float, default=0.1)
    parser.add_argument("--width-threshold", type=float, default=4.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <finetune-dir>/checkpoint_evaluation.csv).",
    )
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("model_epoch_"))
    except ValueError as error:
        raise ValueError(f"Invalid epoch checkpoint name: {path.name}") from error


def collect_models(source_run: Path, finetune_dir: Path):
    original_path = source_run / "best_model.pt"
    if not original_path.is_file():
        raise FileNotFoundError(f"Original model not found: {original_path}")

    checkpoints_dir = finetune_dir / "checkpoints"
    epoch_paths = sorted(
        checkpoints_dir.glob("model_epoch_*.pt"),
        key=checkpoint_epoch,
    )
    # Epoch 0 is a copy of the original model saved before fine-tuning.
    epoch_paths = [path for path in epoch_paths if checkpoint_epoch(path) > 0]
    if not epoch_paths:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {checkpoints_dir}. "
            "Run the updated finetune_microvessels.py first."
        )

    return [("original", 0, original_path)] + [
        ("finetuned", checkpoint_epoch(path), path)
        for path in epoch_paths
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    source_run = Path(args.source_run)
    finetune_dir = Path(args.finetune_dir)
    output_path = (
        Path(args.output)
        if args.output
        else finetune_dir / "checkpoint_evaluation.csv"
    )

    device = select_device(args.device)
    use_amp = device.type == "cuda"
    splits = load_samples(source_run / "splits.csv")
    loaders = {
        split: make_loader(
            splits[split],
            args.batch_size,
            args.workers,
            device,
        )
        for split in args.splits
    }
    models = collect_models(source_run, finetune_dir)
    criterion = ThinWeightedBCEDiceLoss()
    rows = []

    print(f"Device: {device}")
    print(f"Models: {len(models)}")
    print(
        "Evaluation datasets: "
        + ", ".join(
            f"{split} ({len(splits[split])} samples)"
            for split in args.splits
        )
    )

    for model_type, epoch, checkpoint_path in models:
        model, metadata = load_model(checkpoint_path, device)
        saved_epoch = metadata.get("epoch")
        if model_type == "finetuned" and saved_epoch != epoch:
            raise ValueError(
                f"Epoch mismatch for {checkpoint_path}: "
                f"filename={epoch}, checkpoint={saved_epoch}"
            )

        print(f"\n[{model_type}] epoch {epoch}: {checkpoint_path}")
        for split, loader in loaders.items():
            metrics = evaluate_finetune(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                metric_threshold=args.metric_threshold,
                width_threshold=args.width_threshold,
            )
            row = {
                "model_type": model_type,
                "epoch": epoch,
                "saved_epoch": saved_epoch,
                "split": split,
                "sample_count": len(splits[split]),
                "checkpoint": str(checkpoint_path.resolve()),
                **{key: metrics[key] for key in METRIC_KEYS},
            }
            rows.append(row)
            print(f"  {split}: {format_finetune_metrics(metrics)}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_path, rows)
    settings_path = output_path.with_suffix(".json")
    with settings_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_run": str(source_run.resolve()),
                "finetune_dir": str(finetune_dir.resolve()),
                "splits": list(args.splits),
                "metric_threshold": args.metric_threshold,
                "width_threshold": args.width_threshold,
                "model_count": len(models),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved results: {output_path}")
    print(f"Saved settings: {settings_path}")


if __name__ == "__main__":
    main()
