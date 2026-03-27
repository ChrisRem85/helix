import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import csv
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import sys


import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Subset


use_model = int(sys.argv[1])
print (f"Using model config: {use_model}")
def image_transform_fn(pil_img):
    img = np.array(pil_img)              # HWC, uint8
    img = preprocess_input(img)          # HWC, float, encoder-specific norm
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # -> CHW tensor
    return img

mask_transform = transforms.ToTensor()   # no normalization for masks

def build_model(cfg):
    """Return (model, use_model_with_dict_output_flag)"""
    if cfg["kind"] == "smp_unet":
        m = smp.Unet(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return m, False

    if cfg["kind"] == "smp_deeplab":
        m = smp.DeepLabV3Plus(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return m, False

    if cfg["kind"] == "tv_deeplab":
        m = deeplabv3_resnet50(weights=None, num_classes=1)
        return m, True  # returns dict with ["out"]

    raise ValueError(f"Unknown model kind: {cfg['kind']}")

model_configs = [
    # 1) U-Net + ResNet
    {"name": "unet_resnet34", "kind": "smp_unet",   "encoder": "resnet34"},
    # 2) U-Net + EfficientNet
    {"name": "unet_efficientnetb3", "kind": "smp_unet", "encoder": "efficientnet-b3"},
    # 3) U-Net + ViT/MiT (SegFormer encoder)
    {"name": "unet_mit_b2",   "kind": "smp_unet",   "encoder": "mit_b2"},
    # 4) U-Net + larger ViT/MiT
    {"name": "unet_mit_b3",   "kind": "smp_unet",   "encoder": "mit_b3"},
    # 5) DeepLabV3+ (SMP) with ResNet
    {"name": "deeplab_smp_resnet50", "kind": "smp_deeplab", "encoder": "resnet50"},
    # 6) Torchvision DeepLabV3 with ResNet
    {"name": "deeplab_tv_resnet50", "kind": "tv_deeplab", "encoder": "resnet50"},
]


model_cfg = model_configs[use_model]
modelname = f"{model_cfg['name']}.pth"
#modelname = "mit_b2_imagenet_UNet.pth"

model, use_model_with_dict_output = build_model(model_cfg)
preprocess_input = get_preprocessing_fn(model_cfg["encoder"], "imagenet")
"""
for cfg in model_configs:
    model, use_model_with_dict_output = build_model(cfg)

print('loaded each model once to ensure weights are downloaded/cached before training starts.')
quit()
model = smp.Unet(
    encoder_name="mit_b2",       # or "mit_b0", "mit_b3", etc.
    encoder_weights="imagenet",  # uses timm pretrained weights
    in_channels=3,
    classes=1,                   # 1 channel logits for BCEWithLogitsLoss
)

from monai.networks.nets import UNet

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)
model = smp.Unet(
    encoder_name="resnet34",      # or "efficientnet-b0", etc.
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,                    # 1 output channel for BCEWithLogitsLoss
)
"""
"""
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(weights=None, num_classes=1)
use_model_with_dict_output = False  # set to False if your model doesn't return dicts
"""
 # instead of outputs = model(images)
class SkinDataset(Dataset):
    def __init__(self, base_dir, image_transform=image_transform_fn, mask_transform=mask_transform):
        self.base_dir = base_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # store (image_name, mask_name) pairs
        self.samples = []
        skipped = 0

        for f in os.listdir(base_dir):
            name_low = f.lower()

            # use only base images (not mask files)
            if not name_low.endswith(".jpg"):
                continue
            if name_low.endswith("_mask.jpg"):
                continue  # skip mask files themselves

            base = f[:-4]  # strip ".jpg"
            cand_masks = [
                base + "_segmentation.png",
                base + "_mask.jpg",
            ]

            chosen_mask = None
            for m in cand_masks:
                if os.path.exists(os.path.join(base_dir, m)):
                    chosen_mask = m
                    break

            if chosen_mask is None:
                skipped += 1
                print(f"[SKIP] missing mask for {f}, expected one of: {cand_masks}")
                continue

            self.samples.append((f, chosen_mask))

        print(f"[DATASET] usable samples: {len(self.samples)}, skipped: {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, mask_name = self.samples[idx]

        img_path = os.path.join(self.base_dir, img_name)
        mask_path = os.path.join(self.base_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.float32)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # keep returning img_name so your loaders still unpack (images, masks, _)
        return image, mask, img_name

# ==============================
# Transformation
# ==============================

transform = transforms.Compose([
    transforms.ToTensor()
])

# ==============================
# Dataset + Split
# ==============================

base_path = r"./ISIC2018_Task1-2_Training_Input"
fine_path = r"./data_skinstager"
# or:best_val_loss = float("inf")
patience = 5              # number of epochs with no improvement to wait
epochs_no_improve = 0
# base_path = os.path.join(os.getcwd(), "data")

dataset = SkinDataset(base_path)
print(f"Total dataset size (after skipping): {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



#################################################
"""if True:  # for debugging: use only first 64 samples to speed up training
    dataset = SkinDataset(base_path, transform)
    print(f"Total dataset size (after skipping): {len(dataset)}")


    debug_indices = list(range(64))  # first 64 samples
    debug_dataset = Subset(dataset, debug_indices)

    train_size = int(0.8 * len(debug_dataset))
    val_size = len(debug_dataset) - train_size
    train_dataset, val_dataset = random_split(debug_dataset, [train_size, val_size])"""
#####################################################

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

#train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

train_loader = DataLoader(
    train_dataset, 
    batch_size=24, 
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=24, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

# ==============================
# Modell (U-Net)
# ==============================
"""
model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet",
                in_channels=3, classes=1)

model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                          in_channels=3, classes=1)

model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch',
    'unet',
    in_channels=3,
    out_channels=1,
    init_features=32,
    pretrained=True
)
"""
# ==============================
# Training Setup
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
best_val_loss = float("inf")
# ==============================
# Training
# ==============================
os.makedirs("models", exist_ok=True)

metrics_path = os.path.join(
    "models",
    f"{os.path.splitext(modelname)[0]}_metrics.csv"
)

# create / overwrite file and write header
with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phase", "epoch", "train_loss", "val_loss"])

num_epochs = 150
print(f"Using device: {device}")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, masks, _ in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)["out"] if use_model_with_dict_output else model(images)
        # in training/validation loops:

        # Sicherstellen gleiche Shape
        masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ==============================
    # Validation
    # ==============================

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"] if use_model_with_dict_output else model(images)

            masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss_mean = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss:   {val_loss_mean:.4f}")

    # append one row per epoch
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pretrain",               # phase
            epoch + 1,
            train_loss / len(train_loader),
            val_loss_mean,
        ])

    # ----- Early stopping + best model save -----
    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"models/{modelname}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# ==============================
# Modell speichern
# ==============================


# create a folder for models if it doesn't exist
#torch.save(model.state_dict(), f"models/{modelname}")

#print("Modell gespeichert!")

# ==============================
# Validierungsergebnisse speichern
# ==============================

os.makedirs(f"val_preds_{modelname}", exist_ok=True)

with torch.no_grad():
    for images, masks, names in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"] if use_model_with_dict_output else model(images)
        masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

        loss = criterion(outputs, masks)
        val_loss += loss.item()

        # Convert logits -> probabilities -> binary masks
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()  # [B, 1, H, W]

        for pred, name in zip(preds, names):
            pred_np = pred.squeeze(0).cpu().numpy()  # [H, W]
            pred_img = (pred_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(pred_img)

            save_name = f"valid_{name}"
            save_path = os.path.join(f"val_preds_{modelname}", save_name)
            pil_img.save(save_path)

# ==============================
# Fine-tuning dataset (fine_path)
# ==============================
###relode best model before fine-tuning
model.load_state_dict(torch.load(f"models/{modelname}"))

fine_dataset = SkinDataset(fine_path)
print(f"Fine dataset size (after skipping): {len(fine_dataset)}")

fine_train_size = int(0.9 * len(fine_dataset))
fine_val_size = len(fine_dataset) - fine_train_size
fine_train_dataset, fine_val_dataset = random_split(
    fine_dataset, [fine_train_size, fine_val_size]
)

#fine_train_loader = DataLoader(fine_train_dataset, batch_size=24, shuffle=True)
#fine_val_loader = DataLoader(fine_val_dataset, batch_size=24, shuffle=False)

fine_train_loader = DataLoader(
    fine_train_dataset, 
    batch_size=24, 
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

fine_val_loader = DataLoader(
    fine_val_dataset, 
    batch_size=24, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

print(f"Fine Train: {len(fine_train_dataset)}, Fine Val: {len(fine_val_dataset)}")

optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Example: weight positives 3x higher than negatives
# Estimate class imbalance on fine-tuning train set
pos_pixels = 0.0
total_pixels = 0.0

for _, masks, _ in fine_train_loader:
    # masks shape [B, 1, H, W] with values 0 or 1
    pos_pixels += masks.sum().item()
    total_pixels += masks.numel()

pos_fraction = pos_pixels / max(total_pixels, 1e-6)
neg_fraction = 1.0 - pos_fraction

# Standard choice: pos_weight = neg / pos
pos_weight_value = neg_fraction / max(pos_fraction, 1e-6)
print(f"Estimated pos_weight for fine-tuning: {pos_weight_value:.3f}")

#pos_weight_value = 3.0  # tune this
pos_weight = torch.tensor([pos_weight_value], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

fine_epochs = 30
best_val_loss_fine = float("inf")
epochs_no_improve_fine = 0
patience_fine = 5

for epoch in range(fine_epochs):
    model.train()
    train_loss = 0.0

    for images, masks, _ in tqdm(fine_train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"] if use_model_with_dict_output else model(images)

        masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, _ in fine_val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"] if use_model_with_dict_output else model(images)
            masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss_mean = val_loss / len(fine_val_loader)
    print(f"\n[Fine] Epoch {epoch+1}/{fine_epochs}")
    print(f"[Fine] Train Loss: {train_loss/len(fine_train_loader):.4f}")
    print(f"[Fine] Val Loss:   {val_loss_mean:.4f}")

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "finetune",
            epoch + 1,
            train_loss / len(fine_train_loader),
            val_loss_mean,
        ])

    if val_loss_mean < best_val_loss_fine:
        best_val_loss_fine = val_loss_mean
        epochs_no_improve_fine = 0
        torch.save(model.state_dict(), f"models/{modelname}_ft.pth")
    else:
        epochs_no_improve_fine += 1
        if epochs_no_improve_fine >= patience_fine:
            print(f"[Fine] Early stopping after {epoch+1} epochs.")
            break