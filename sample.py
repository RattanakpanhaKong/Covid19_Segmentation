import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Define the building blocks of UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Define the dataset
class XRayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Training function
def train_model(image_dir, mask_dir, epochs=5, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = XRayDataset(image_dir, mask_dir, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss_values = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f'Epoch {epoch + 1}, Iteration {batch_idx + 1}, Loss: {loss.item()}')

        loss_values.append(epoch_loss / len(dataloader))

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')
        print("=====================================================================")

    torch.save(model.state_dict(), 'unet_model.pth')

    plt.plot(range(1, epochs + 1), loss_values, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# IoU and mIoU calculations
def iou(pred, target, n_classes=1):
    smooth = 1e-6  # To avoid division by zero
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred & target).sum((2, 3))
    union = (pred | target).sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def evaluate_model(model, dataloader, device):
    model.eval()
    iou_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            iou_total += iou(outputs, masks).item()
            n_batches += 1

    mean_iou = iou_total / n_batches
    print(f'Mean IoU: {mean_iou:.4f}')

if __name__ == "__main__":
    print("Start training")
    train_model('data/train/image', 'data/train/mask')
    print("Finished Training")

    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load('unet_model.pth'))
    model.eval()

    test_dataset = XRayDataset('data/test/image', 'data/test/mask', transform=get_transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate_model(model, test_dataloader, device)
