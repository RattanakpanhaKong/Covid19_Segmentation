import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data

# Base directory path
base_directory = 'CHENDA_CHOUN/Brain_tumor_segmentation/Brats_dataset/'

# Define paths to train, valid, and test directories
train_dir = os.path.join(base_directory, 'train')
valid_dir = os.path.join(base_directory, 'valid')
test_dir = os.path.join(base_directory, 'test')

def list_dir_contents(directory):
    for root, dirs, files in os.walk(directory):
        print(f"Root: {root}")
        for d in dirs:
            print(f"  Directory: {d}")
        for f in files:
            print(f"  File: {f}")

# Print contents of train, valid, and test directories
list_dir_contents(train_dir)
list_dir_contents(valid_dir)
list_dir_contents(test_dir)

# Load a sample image for plotting
sample_image = np.load(os.path.join(train_dir, 'BraTS19_2013_15_1/img.npy'))

def plot_image(sample_image, num_image=4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    channel_names = ["1st channel image", "2nd channel image", "3rd channel image", "4th channel image"]
    for i in range(num_image):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(sample_image[i], cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(channel_names[i])
    plt.show()

plot_image(sample_image)

class BratsDataset_seg(torch.utils.data.Dataset):
    def __init__(self, root, learning_by_channel):
        self.root = root
        print("Dataset root directory:", self.root)
        self.data_list = list(sorted(os.listdir(self.root)))
        self.learning_by_channel = learning_by_channel

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data_list[idx], 'img.npy')
        label_path = os.path.join(self.root, self.data_list[idx], 'label.npy')

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"File missing for index {idx}: img path - {img_path}, label path - {label_path}")
            return None

        img = torch.Tensor(np.load(img_path))
        label = np.load(label_path)

        if self.learning_by_channel:
            label[label == 4] = 3
            label = torch.Tensor(label).type(torch.LongTensor)
            output = {'img': img, 'label': label}
        else:
            label[label > 0] = 1
            label = torch.Tensor(label).type(torch.LongTensor)
            output = {'img': img, 'label': label}

        return output

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Test dataset loading
train_dataset = BratsDataset_seg(train_dir, learning_by_channel=False)
valid_dataset = BratsDataset_seg(valid_dir, learning_by_channel=False)
test_dataset = BratsDataset_seg(test_dir, learning_by_channel=False)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4, collate_fn=collate_fn)

print("Sample data item shapes:")
print("Image shape:", train_dataset[0]['img'].shape)
print("Label shape:", train_dataset[0]['label'].shape)
plt.imshow(train_dataset[0]['label'])

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
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

def train_one_epoch(model, optimizer, criterion, train_loader, valid_loader, device, epoch, lr_scheduler, print_freq=30, min_valid_loss=float('inf')):
    model.train()
    running_loss = 0.0
    for batch_idx, pack in enumerate(train_loader):
        if pack is None:
            continue  # Skip empty batches
        inputs = pack['img'].to(device)
        labels = pack['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % print_freq == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for valid_iter, pack in enumerate(valid_loader):
            if pack is None:
                continue  # Skip empty batches
            inputs = pack['img'].to(device)
            labels = pack['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(valid_loader)
    print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}')

    if avg_val_loss < min_valid_loss:
        min_valid_loss = avg_val_loss
        torch.save(model.state_dict(), 'CHENDA_CHOUN/Brain_tumor_segmentation/best_model_Chenda.pth')
        print('Model saved!')

    lr_scheduler.step()
    return min_valid_loss

def evaluate(model, test_data_loader, device, model_type, sklearn=None):
    model.eval()
    if model_type == 'seg_channel':
        iou_list = np.zeros((4,))
        for iter, pack in enumerate(test_data_loader):
            if pack is None:
                continue  # Skip empty batches
            img = pack['img'].to(device)
            label = pack['label']
            pred = model(img)
            pred = pred.cpu().detach().numpy()
            (B, C, W, H) = pred.shape
            pred = np.argmax(pred[0], axis=0)
            for num_channel in range(C):
                intersection = np.logical_and(label[0, num_channel], pred == num_channel).sum()
                union = np.logical_or(label[0, num_channel], pred == num_channel).sum()
                iou = (intersection + 1e-10) / (union + 1e-10)
                iou_list[num_channel] += iou

        ious = []
        for channel in range(len(iou_list)):
            zero_index = np.where(iou_list[channel] == 0)
            miou = np.delete(iou_list[channel], zero_index)
            print('{} channel miou: {}'.format(channel, miou / len(test_data_loader)))
            ious.append(miou / len(test_data_loader))
        print(f"average iou : {np.mean(ious[1:])}")

    elif model_type == 'clf':
        accuracy = 0
        pred_list = []
        label_list = []
        for iter, pack in enumerate(test_data_loader):
            if pack is None:
                continue  # Skip empty batches
            img = pack['img'].to(device)
            label = pack['label']
            pred = model(img)
            pred = pred.cpu().detach().numpy()
            if pred.shape[1] == 2:
                pred_list.append(np.argmax(pred))
            elif pred.shape[1] == 1:
                pred_list.append(np.round(pred[0]))
            label_list.append(label)

        if pred.shape[1] == 1:
            pred_list = np.concatenate(pred_list, axis=0).astype(np.int)
            label_list = [np.array(e) for e in label_list]
            label_list = np.concatenate(np.array(label_list), axis=0)
            from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
            print('accuracy: {} recall: {} precision: {} f1 score: {}'.format(
            accuracy_score(label_list, pred_list),
            recall_score(label_list, pred_list),
            precision_score(label_list, pred_list),
            f1_score(label_list, pred_list)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 1
n_channels = 4
n_classes = 4

model_channel = UNet(n_channels=n_channels, n_classes=n_classes)
model_channel.to(device)

optimizer = torch.optim.Adam(model_channel.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=111, gamma=1)

train_dataset = BratsDataset_seg(train_dir, learning_by_channel=True)
valid_dataset = BratsDataset_seg(valid_dir, learning_by_channel=True)
test_dataset = BratsDataset_seg(test_dir, learning_by_channel=True)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4, collate_fn=collate_fn)

min_val_loss = 200
for epoch in range(num_epochs):
    min_val_loss = train_one_epoch(model_channel, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler, print_freq=30, min_valid_loss=min_val_loss)
    print('validation...')
    evaluate(model_channel, valid_data_loader, device=device, model_type='seg_channel')
print('testing the lastly updated model')
evaluate(model_channel, test_data_loader, device=device, model_type = 'seg_channel')

## test data visualize
pack = test_dataset[1]
img = torch.Tensor(np.expand_dims(pack['img'], 0)).to(device)
label = pack['label'].to(device)
pred = model_channel(img)
pred = pred.cpu().detach().numpy()

img_np = img.cpu().detach().numpy()[0, 0, :, :]  # Assuming the image is grayscale
label_np = label.cpu().detach().numpy()
pred_np = pred[0, 0, :, :]  # Assuming the prediction is also grayscale

# Visualize the original image, the ground truth label, and the predicted segmentation

def plot_image(sample_image, num_image=4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    channel_names = ["1st channel image", "2nd channel image", "3rd channel image", "4th channel image"]
    for i in range(num_image):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(sample_image[i], cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(channel_names[i])
    plt.show()


def visualize_prediction(image, label, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Display original image (using the first channel for simplicity)
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display ground truth label
    axes[1].imshow(label, cmap='viridis', alpha=0.5)
    axes[1].set_title('Ground Truth Label')
    axes[1].axis('off')

    # Display predicted segmentation
    axes[2].imshow(image[0], cmap='gray')
    axes[2].imshow(prediction, cmap='viridis', alpha=0.5)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    plt.show()


# Test data visualize
pack = test_dataset[1]
img = torch.Tensor(np.expand_dims(pack['img'], 0)).to(device)
label = pack['label'].to(device)
pred = model_channel(img)
pred = pred.cpu().detach().numpy()
pred = np.argmax(pred, axis=1)[0]  # Get the predicted segmentation map

img_np = img.cpu().detach().numpy()[0]  # Assuming the image has multiple channels
label_np = label.cpu().detach().numpy()
pred_np = pred  # The predicted segmentation map

# Visualize the original image, the ground truth label, and the predicted segmentation
plot_image(img_np)
visualize_prediction(img_np, label_np, pred_np)
