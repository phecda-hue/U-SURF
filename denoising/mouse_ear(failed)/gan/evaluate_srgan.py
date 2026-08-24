import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset, random_split

from visualize_srgan_sample import (
    DEFAULT_DATA_ROOT,
    FRDGenerator,
    SCALE_FACTOR,
    SuperResolutionPairDataset,
    clean_state_dict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "gan_checkpoints" / "best_generator.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quickly compare SRGAN against bicubic on a fixed validation subset."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_generator(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("generator", checkpoint)
    model = FRDGenerator(scale_factor=SCALE_FACTOR).to(device)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model, checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


def image_metrics(prediction: np.ndarray, target: np.ndarray):
    difference = prediction - target
    mse = float(np.mean(difference * difference))
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    ssim = float(structural_similarity(target, prediction, data_range=1.0))
    mae = float(np.mean(np.abs(difference)))
    return psnr, ssim, mae


def main():
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("--samples and --batch-size must be >= 1.")

    device = torch.device(args.device)
    dataset = SuperResolutionPairDataset(data_root=args.data_root)
    validation_size = max(1, int(len(dataset) * 0.1))
    training_size = len(dataset) - validation_size
    _, validation = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    sample_count = min(args.samples, len(validation))
    validation = Subset(validation, range(sample_count))
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    model, saved_epoch = load_generator(Path(args.checkpoint), device)

    sr_metrics = []
    bicubic_metrics = []
    inference_seconds = 0.0
    with torch.inference_mode():
        for low_resolution, high_resolution in loader:
            low_resolution = low_resolution.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            super_resolved = model(low_resolution).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            bicubic = F.interpolate(
                low_resolution,
                size=high_resolution.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
            sr_numpy = super_resolved[:, 0].cpu().numpy()
            bicubic_numpy = bicubic[:, 0].cpu().numpy()
            hr_numpy = high_resolution[:, 0].numpy()
            for sr_image, bicubic_image, hr_image in zip(
                sr_numpy, bicubic_numpy, hr_numpy
            ):
                sr_metrics.append(image_metrics(sr_image, hr_image))
                bicubic_metrics.append(image_metrics(bicubic_image, hr_image))

    sr = np.asarray(sr_metrics)
    bicubic = np.asarray(bicubic_metrics)
    names = ("PSNR", "SSIM", "MAE")
    print(f"Checkpoint: {args.checkpoint} (saved epoch={saved_epoch})")
    print(f"Validation samples: {sample_count}/{validation_size}, device={device}")
    print("\nMean metric comparison")
    for index, name in enumerate(names):
        difference = sr[:, index].mean() - bicubic[:, index].mean()
        print(
            f"  {name:4s}: SRGAN={sr[:, index].mean():.6f}, "
            f"bicubic={bicubic[:, index].mean():.6f}, delta={difference:+.6f}"
        )
    psnr_wins = int(np.count_nonzero(sr[:, 0] > bicubic[:, 0]))
    ssim_wins = int(np.count_nonzero(sr[:, 1] > bicubic[:, 1]))
    mae_wins = int(np.count_nonzero(sr[:, 2] < bicubic[:, 2]))
    print("\nSRGAN wins over bicubic")
    print(f"  PSNR: {psnr_wins}/{sample_count}")
    print(f"  SSIM: {ssim_wins}/{sample_count}")
    print(f"  MAE:  {mae_wins}/{sample_count}")
    print(
        f"\nInference: {1000.0 * inference_seconds / sample_count:.2f} ms/image "
        f"({sample_count / inference_seconds:.2f} images/s)"
    )


if __name__ == "__main__":
    main()
