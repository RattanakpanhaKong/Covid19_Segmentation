import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import XRayDataset, get_transforms
from unet import UNet
from utils import dice_loss


def train_model(image_dir, mask_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = XRayDataset(image_dir, mask_dir, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Ensure the dimensions match
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            # loss = dice_loss(outputs, masks)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')

    torch.save(model.state_dict(), 'unet_model.pth')




if __name__ == "__main__":
    print("Start training")
    train_model('data/train/image', 'data/train/mask')
    print("Finished Training")
