import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

num_epochs = 500
ROOT = Path(__file__).resolve().parents[1]
PA_SIMULATIONS_ROOT = ROOT / "data" / "pa simulations"
PA_FIRST_PAGE = 20  # One-based, inclusive
PA_LAST_PAGE = 100  # One-based, inclusive
IMAGE_SIZE = 256
SCALE_FACTOR = 4
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
SPLIT_SEED = 42
CHECKPOINT_DIR = ROOT / "gan_checkpoints"
CHECKPOINT_EVERY = 50
PATIENCE = 30


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    group: str
    frame: int | None = None


def collect_pa_simulation_records(data_root: Path) -> list[ImageRecord]:
    """Index pages 20-100 of every PA TIFF stack (PIL indices 19-99)."""
    tiff_paths = sorted((*data_root.rglob("*.tif"), *data_root.rglob("*.tiff")))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIF/TIFF files found in {data_root}")
    records: list[ImageRecord] = []
    for path in tiff_paths:
        with Image.open(path) as image:
            frame_count = getattr(image, "n_frames", 1)
        if frame_count < PA_LAST_PAGE:
            raise ValueError(
                f"Expected at least {PA_LAST_PAGE} frames, but {path} has {frame_count}"
            )
        group = f"pa:{path.relative_to(data_root).as_posix()}"
        records.extend(
            ImageRecord(path, group, frame)
            for frame in range(PA_FIRST_PAGE - 1, PA_LAST_PAGE)
        )
    return records


def split_records_by_group(
    records: list[ImageRecord], validation_split: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Keep all selected pages of a TIFF in the same train/validation split."""
    groups = sorted({record.group for record in records})
    random.Random(seed).shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * validation_split))])
    train = [record for record in records if record.group not in validation_groups]
    validation = [record for record in records if record.group in validation_groups]
    return train, validation


class SuperResolutionPairDataset(Dataset):
    """Create LR/HR pairs from individual pages of PA TIFF stacks."""

    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int = IMAGE_SIZE,
        scale_factor: int = SCALE_FACTOR,
    ) -> None:
        if not records:
            raise FileNotFoundError("No training images were found")
        self.records = records
        self.image_size = image_size
        self.scale_factor = scale_factor
        if image_size % scale_factor != 0:
            raise ValueError("image_size must be divisible by scale_factor")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_grayscale(record: ImageRecord) -> np.ndarray:
        path = record.path
        with Image.open(path) as image:
            if record.frame is not None:
                image.seek(record.frame)
            array = np.asarray(image).copy()
        if array.ndim != 2:
            raise ValueError(f"Expected grayscale image: {path}")
        return array

    @staticmethod
    def _normalize(array: np.ndarray, path: Path) -> np.ndarray:
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        minimum, maximum = float(array.min()), float(array.max())
        if minimum >= 0.0 and maximum <= 1.0:
            return array
        if minimum >= 0.0 and maximum <= 255.0:
            return array / 255.0
        if minimum >= 0.0 and maximum <= 65535.0:
            return array / 65535.0
        raise ValueError(f"Unsupported float intensity range [{minimum}, {maximum}]: {path}")

    def _pad_or_resize(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape
        if height <= self.image_size and width <= self.image_size:
            top = (self.image_size - height) // 2
            bottom = self.image_size - height - top
            left = (self.image_size - width) // 2
            right = self.image_size - width - left
            return np.pad(array, ((top, bottom), (left, right)), constant_values=0)

        image = Image.fromarray(array)
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.asarray(image).copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        raw_image = self._read_grayscale(record)
        hr_image = self._normalize(raw_image, record.path)
        hr_image = self._pad_or_resize(hr_image)
        hr_tensor = torch.from_numpy(hr_image).unsqueeze(0).clamp(0.0, 1.0)

        lr_size = self.image_size // self.scale_factor
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0),
            size=(lr_size, lr_size),
            mode="area",
        ).squeeze(0)

        return lr_tensor, hr_tensor


all_records = collect_pa_simulation_records(PA_SIMULATIONS_ROOT)
train_records, val_records = split_records_by_group(all_records, VALIDATION_SPLIT, SPLIT_SEED)
train_dataset = SuperResolutionPairDataset(train_records, IMAGE_SIZE, SCALE_FACTOR)
val_dataset = SuperResolutionPairDataset(val_records, IMAGE_SIZE, SCALE_FACTOR)
print(
    f"Dataset: train={len(train_dataset):,}, validation={len(val_dataset):,}, "
    f"TIFF files={len({record.path for record in all_records}):,}, "
    f"frames per TIFF={PA_LAST_PAGE - PA_FIRST_PAGE + 1} "
    f"(pages {PA_FIRST_PAGE}-{PA_LAST_PAGE})"
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class FRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = channels + i * growth_channels

            self.layers.append(
                nn.Sequential(
                    DSConv(in_ch, growth_channels, kernel_size=3),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Local Feature Fusion
        self.lff = nn.Conv2d(
            channels + num_layers * growth_channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            concat_features = torch.cat(features, dim=1)
            out = layer(concat_features)
            features.append(out)

        fused = torch.cat(features, dim=1)
        local_feature = self.lff(fused)

        # Local residual learning
        return local_feature + x
    
class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()

        self.conv = DSConv(
            channels,
            channels * (scale_factor ** 2),
            kernel_size=3
        )

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        return x
    
class FRDGenerator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=64,
        growth_channels=32,
        num_frdb=6,
        num_layers_per_frdb=4,
        scale_factor=4
    ):
        super().__init__()

        self.shallow = nn.Sequential(
            DSConv(in_channels, channels, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.frdbs = nn.ModuleList([
            FRDB(
                channels=channels,
                growth_channels=growth_channels,
                num_layers=num_layers_per_frdb
            )
            for _ in range(num_frdb)
        ])

        # Dense Feature Fusion
        self.global_fusion = nn.Sequential(
            nn.Conv2d(num_frdb * channels, channels, kernel_size=1),
            DSConv(channels, channels, kernel_size=3)
        )

        upsample_blocks = []

        if scale_factor == 2:
            upsample_blocks.append(UpsampleBlock(channels, 2))
        elif scale_factor == 4:
            upsample_blocks.append(UpsampleBlock(channels, 2))
            upsample_blocks.append(UpsampleBlock(channels, 2))
        elif scale_factor == 8:
            upsample_blocks.append(UpsampleBlock(channels, 2))
            upsample_blocks.append(UpsampleBlock(channels, 2))
            upsample_blocks.append(UpsampleBlock(channels, 2))
        else:
            raise ValueError("scale_factor는 2, 4, 8 중 하나를 권장합니다.")

        self.upsample = nn.Sequential(*upsample_blocks)

        self.reconstruction = nn.Sequential(
            DSConv(channels, out_channels, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        shallow_feature = self.shallow(x)

        frdb_outputs = []
        out = shallow_feature

        for frdb in self.frdbs:
            out = frdb(out)
            frdb_outputs.append(out)

        fused = torch.cat(frdb_outputs, dim=1)
        fused = self.global_fusion(fused)

        # Global residual connection
        out = fused + shallow_feature

        out = self.upsample(out)
        out = self.reconstruction(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, use_dsconv=False):
        super().__init__()

        def conv_block(in_ch, out_ch, stride, use_bn=True):
            Conv = DSConv if use_dsconv else nn.Conv2d

            if use_dsconv:
                conv = Conv(in_ch, out_ch, kernel_size=3, stride=stride)
            else:
                conv = Conv(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

            layers = [conv]

            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(in_channels, 64, stride=1, use_bn=False),
            conv_block(64, 64, stride=2),

            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),

            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),

            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
from torchvision.models import vgg19, VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # PyTorch VGG19 기준
        # conv2_2/relu2_2 근처: [:8]
        # conv5_4/relu5_4 근처: [:35]
        self.vgg_2_2 = nn.Sequential(*list(vgg.children())[:8])
        self.vgg_5_4 = nn.Sequential(*list(vgg.children())[:35])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, x):
        # 광음향 영상이 1채널이면 VGG 입력을 위해 3채널로 복제
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # x는 [0, 1] 범위라고 가정
        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        x = self.preprocess(x)

        feat_2_2 = self.vgg_2_2(x)
        feat_5_4 = self.vgg_5_4(x)

        return feat_2_2, feat_5_4
    
class FRDGANLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=0.01,
        pixel_weight=1.0,
        adv_weight=1e-3,
    ):
        super().__init__()

        self.vgg = VGGFeatureExtractor()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

        self.alpha = alpha
        self.beta = beta
        self.pixel_weight = pixel_weight
        self.adv_weight = adv_weight

    def content_loss(self, sr, hr):
        sr_2_2, sr_5_4 = self.vgg(sr)

        with torch.no_grad():
            hr_2_2, hr_5_4 = self.vgg(hr)

        loss_5_4 = self.mse(sr_5_4, hr_5_4)
        loss_2_2 = self.mse(sr_2_2, hr_2_2)

        return (
            self.alpha * loss_5_4
            + self.beta * loss_2_2
        )

    def pretrain_loss(self, sr, hr): return self.l1(sr, hr)
    '''pixel = self.l1(sr, hr)
        content = self.content_loss(sr, hr)

        return pixel + content
'''
    def generator_loss(self, sr, hr, fake_logits):
        # 1. 혈관 구조를 정확한 위치에 복원
        pixel = weighted_l1_loss(
            sr,
            hr,
            threshold=0.02,
            signal_weight=10.0
        )

        # 2. 전체적인 특징 보존
        content = self.content_loss(sr, hr)

        # 3. 자연스러운 세부 구조
        adv = self.bce(
            fake_logits,
            torch.ones_like(fake_logits)
        )

        total = (
            pixel
            + 0.1 * content
            + self.adv_weight * adv
        )

        return total, pixel, content, adv

    # GAN 학습에서 Discriminator용 loss
    def discriminator_loss(self, real_logits, fake_logits):
        # label smoothing 적용
        real_labels = torch.full_like(real_logits, 0.9)
        fake_labels = torch.full_like(fake_logits, 0.1)

        real_loss = self.bce(real_logits, real_labels)
        fake_loss = self.bce(fake_logits, fake_labels)

        return 0.5 * (real_loss + fake_loss)
    
    
    
def train_step(
    lr_img,
    hr_img,
    generator,
    discriminator,
    criterion,
    optimizer_g,
    optimizer_d,
    device,
    step,
    d_interval=5
):
    lr_img = lr_img.to(device)
    hr_img = hr_img.to(device)

    # 1. Train Discriminator less often
    if step % d_interval == 0:
        optimizer_d.zero_grad()

        with torch.no_grad():
            sr_img = generator(lr_img)

        real_logits = discriminator(hr_img)
        fake_logits = discriminator(sr_img.detach())

        loss_d = criterion.discriminator_loss(real_logits, fake_logits)
        loss_d.backward()
        optimizer_d.step()

        real_prob = torch.sigmoid(real_logits).mean().item()
        fake_prob_d = torch.sigmoid(fake_logits).mean().item()
    else:
        loss_d = torch.tensor(0.0, device=device)
        real_prob = 0.0
        fake_prob_d = 0.0

    # 2. Train Generator every step
    optimizer_g.zero_grad()

    sr_img = generator(lr_img)
    fake_logits_g = discriminator(sr_img)

    loss_g, loss_pixel, loss_content, loss_adv = criterion.generator_loss(
        sr_img,
        hr_img,
        fake_logits_g
    )

    loss_g.backward()
    optimizer_g.step()

    fake_prob_g = torch.sigmoid(fake_logits_g).mean().item()

    return {
        "loss_g": loss_g.item(),
        "loss_pixel": loss_pixel.item(),
        "loss_d": loss_d.item(),
        "loss_content": loss_content.item(),
        "loss_adv": loss_adv.item(),
        "real_prob": real_prob,
        "fake_prob_g": fake_prob_g,
        "d_updated": step % d_interval == 0,
    }


def model_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def evaluate_generator(
    generator: nn.Module,
    val_loader: DataLoader,
    criterion: FRDGANLoss,
    device: str,
) -> float:
    was_training = generator.training
    generator.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr_img = generator(lr_img)
            loss = criterion.pretrain_loss(sr_img, hr_img)

            batch_size = lr_img.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if was_training:
        generator.train()

    return total_loss / max(total_samples, 1)


def save_training_checkpoint(
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    val_loss: float,
    best_val_loss: float,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "generator": model_state_dict(generator),
            "discriminator": model_state_dict(discriminator),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


device = "cuda" if torch.cuda.is_available() else "cpu"

generator = FRDGenerator(
    in_channels=1,
    out_channels=1,
    channels=64,
    growth_channels=32,
    num_frdb=6,
    num_layers_per_frdb=4,
    scale_factor=SCALE_FACTOR
).to(device)

discriminator = Discriminator(
    in_channels=1,
    use_dsconv=False
).to(device)

criterion = FRDGANLoss(
    alpha=1.0,
    beta=0.01,
    pixel_weight=1.0,
    adv_weight=1e-3
).to(device)

optimizer_g = torch.optim.Adam(
    generator.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999)
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=1e-6,
    betas=(0.5, 0.999)
)
pretrain_epochs = 50
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def weighted_l1_loss(sr, hr, threshold=0.02, signal_weight=10.0):
    signal_mask = (hr > threshold).float()

    weight = (
        1.0
        + signal_mask * (signal_weight - 1.0)
    )

    return torch.mean(
        weight * torch.abs(sr - hr)
    )

for epoch in range(pretrain_epochs):
    generator.train()

    total_loss = 0.0

    for lr_img, hr_img in train_loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        optimizer_g.zero_grad()

        sr_img = generator(lr_img)
        'loss = criterion.pretrain_loss(sr_img, hr_img)'
        loss = weighted_l1_loss(
            sr_img,
            hr_img
        )

        loss.backward()
        optimizer_g.step()

        total_loss += loss.item()

    print(f"[Pretrain {epoch+1}/{pretrain_epochs}] G_pre: {total_loss / len(train_loader):.4f}")

best_val_loss = evaluate_generator(generator, val_loader, criterion, device)
counter = 0
torch.save(model_state_dict(generator), CHECKPOINT_DIR / "best_generator.pth")
print(f"Initial validation loss after pretrain: {best_val_loss:.4f}")

global_step = 0

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    sum_g = 0.0
    sum_pixel = 0.0
    sum_content = 0.0
    sum_adv = 0.0
    sum_fake_prob = 0.0

    sum_d = 0.0
    sum_real_prob = 0.0
    d_count = 0

    for lr_img, hr_img in train_loader:
        logs = train_step(
            lr_img,
            hr_img,
            generator,
            discriminator,
            criterion,
            optimizer_g,
            optimizer_d,
            device,
            global_step,
            d_interval=5
        )

        global_step += 1

        sum_g += logs["loss_g"]
        sum_pixel += logs["loss_pixel"]
        sum_content += logs["loss_content"]
        sum_adv += logs["loss_adv"]
        sum_fake_prob += logs["fake_prob_g"]

        if logs["d_updated"]:
            sum_d += logs["loss_d"]
            sum_real_prob += logs["real_prob"]
            d_count += 1

    n = len(train_loader)

    avg_d = sum_d / max(d_count, 1)
    avg_real_prob = sum_real_prob / max(d_count, 1)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"G: {sum_g/n:.4f} "
        f"D: {avg_d:.4f} "
        f"Pixel: {sum_pixel/n:.4f} "
        f"Content: {sum_content/n:.4f} "
        f"Adv: {sum_adv/n:.4f} "
        f"RealProb: {avg_real_prob:.3f} "
        f"FakeProb: {sum_fake_prob/n:.3f}"
    )

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = save_training_checkpoint(
            epoch=epoch + 1,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            val_loss=best_val_loss,
            best_val_loss=best_val_loss,
        )
        print(f"Saved checkpoint: {checkpoint_path}")
