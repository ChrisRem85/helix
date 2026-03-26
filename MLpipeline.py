import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SkinDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform

        self.image_names = [
            f for f in os.listdir(base_dir)
            if f.endswith(".jpg") and not f.endswith("_mask.jpg")
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        mask_name = img_name.replace(".jpg", "_mask.jpg")

        img_path = os.path.join(self.base_dir, img_name)
        mask_path = os.path.join(self.base_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))

        mask = (mask > 0).astype(np.float32)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# ==============================
# Transformation
# ==============================

transform = transforms.Compose([
    transforms.ToTensor()
])

# ==============================
# Dataset + Split
# ==============================

base_path = "/Users/lisahaut/Desktop/ML_Hackathon/SkinStager-Final für Teilnehmer/MM1-dorsal"

dataset = SkinDataset(base_path, transform)



train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ==============================
# Modell (U-Net)
# ==============================

model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch',
    'unet',
    in_channels=3,
    out_channels=1,
    init_features=32,
    pretrained=True
)

# ==============================
# Training Setup
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ==============================
# Training
# ==============================

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

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
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss:   {val_loss/len(val_loader):.4f}")

# ==============================
# Modell speichern
# ==============================

torch.save(model.state_dict(), "skin_unet_model.pth")

print("Modell gespeichert!")
