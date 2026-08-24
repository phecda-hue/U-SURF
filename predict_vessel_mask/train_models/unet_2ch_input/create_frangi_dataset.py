import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import frangi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "augmented_training_dataset_3x"
DEFAULT_OUTPUT = ROOT / "data" / "augmented_training_dataset_3x_frangi_sigma_2_20"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute Frangi images for original+Frangi 2-channel training."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sigma-min", type=int, default=2)
    parser.add_argument("--sigma-max", type=int, default=20)
    parser.add_argument("--normalization-percentile", type=float, default=99.9)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute Frangi files that already exist.",
    )
    return parser.parse_args()


def read_uint16(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Expected an integer grayscale image: {path}")
    return array.astype(np.uint16, copy=False)


def create_frangi_image(task):
    source_path, output_path, sigma_min, sigma_max, percentile, overwrite = task
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.is_file() and not overwrite:
        return source_path.name, "skipped"

    image = read_uint16(source_path).astype(np.float32) / 65535.0
    response = frangi(
        image,
        sigmas=range(sigma_min, sigma_max + 1),
        black_ridges=False,
        mode="reflect",
    )
    scale = float(np.percentile(response, percentile))
    if np.isfinite(scale) and scale > 0:
        response = np.clip(response / scale, 0.0, 1.0)
    else:
        response = np.zeros_like(response, dtype=np.float32)
    output = np.round(response * 65535.0).astype(np.uint16)
    Image.fromarray(output, mode="I;16").save(output_path, format="PNG")
    return source_path.name, "created"


def copy_dataset_files(source: Path, output: Path, file_names: list[str]):
    image_dir = output / "images"
    mask_dir = output / "masks"
    frangi_dir = output / "frangi"
    for directory in (image_dir, mask_dir, frangi_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, file_name in enumerate(file_names, start=1):
        source_image = source / "images" / file_name
        source_mask = source / "masks" / file_name
        if not source_image.is_file() or not source_mask.is_file():
            raise FileNotFoundError(
                f"Missing image/mask pair for {file_name}: "
                f"image={source_image.is_file()}, mask={source_mask.is_file()}"
            )
        target_image = image_dir / file_name
        target_mask = mask_dir / file_name
        if not target_image.is_file():
            shutil.copy2(source_image, target_image)
        if not target_mask.is_file():
            shutil.copy2(source_mask, target_mask)
        if index % 200 == 0 or index == len(file_names):
            print(f"Copied original image/mask pairs: {index}/{len(file_names)}")


def write_manifest(source: Path, output: Path, rows: list[dict]):
    fieldnames = list(rows[0].keys()) + [
        "prepared_original_image",
        "prepared_frangi_image",
        "prepared_mask",
    ]
    with (output / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            file_name = row["file_name"]
            writer.writerow(
                {
                    **row,
                    "prepared_original_image": f"images/{file_name}",
                    "prepared_frangi_image": f"frangi/{file_name}",
                    "prepared_mask": f"masks/{file_name}",
                }
            )


def validate_dataset(output: Path, file_names: list[str]):
    for folder in ("images", "frangi", "masks"):
        names = {path.name for path in (output / folder).glob("*.png")}
        expected = set(file_names)
        if names != expected:
            raise RuntimeError(
                f"Validation failed for {folder}: "
                f"missing={len(expected - names)}, extra={len(names - expected)}"
            )

    for file_name in file_names:
        original = read_uint16(output / "images" / file_name)
        filtered = read_uint16(output / "frangi" / file_name)
        with Image.open(output / "masks" / file_name) as mask_image:
            mask = np.asarray(mask_image)
        if original.shape != filtered.shape or original.shape != mask.shape:
            raise RuntimeError(f"Shape mismatch for {file_name}")


def main():
    args = parse_args()
    if args.sigma_min < 1 or args.sigma_max < args.sigma_min:
        raise ValueError("Sigma values must satisfy 1 <= sigma-min <= sigma-max.")
    if not 0 < args.normalization_percentile <= 100:
        raise ValueError("--normalization-percentile must be in (0, 100].")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise ValueError("Source and output directories must be different.")
    manifest_path = source / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")
    file_names = [row["file_name"] for row in rows]
    if len(file_names) != len(set(file_names)):
        raise ValueError("Duplicate file_name values found in the source manifest.")

    output.mkdir(parents=True, exist_ok=True)
    copy_dataset_files(source, output, file_names)
    tasks = [
        (
            str(source / "images" / file_name),
            str(output / "frangi" / file_name),
            args.sigma_min,
            args.sigma_max,
            args.normalization_percentile,
            args.overwrite,
        )
        for file_name in file_names
    ]
    created = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (_, status) in enumerate(executor.map(create_frangi_image, tasks), 1):
            created += status == "created"
            skipped += status == "skipped"
            if index % 25 == 0 or index == len(tasks):
                print(
                    f"Frangi sigma {args.sigma_min}-{args.sigma_max}: "
                    f"{index}/{len(tasks)} (created={created}, skipped={skipped})",
                    flush=True,
                )

    write_manifest(source, output, rows)
    source_info_path = source / "dataset_info.json"
    source_info = {}
    if source_info_path.is_file():
        with source_info_path.open(encoding="utf-8") as file:
            source_info = json.load(file)
    info = {
        **source_info,
        "dataset_type": "two_channel_original_plus_precomputed_frangi",
        "source_dataset": str(source),
        "sample_count": len(file_names),
        "channels": ["original", "frangi"],
        "frangi": {
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "sigmas": list(range(args.sigma_min, args.sigma_max + 1)),
            "black_ridges": False,
            "mode": "reflect",
            "normalization": f"per-image percentile {args.normalization_percentile}",
            "output_dtype": "uint16",
            "output_range": [0, 65535],
        },
    }
    with (output / "dataset_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)

    validate_dataset(output, file_names)
    print(f"Dataset created and validated: {output}")
    print(f"Samples: {len(file_names)}, channels: original + Frangi")


if __name__ == "__main__":
    main()
