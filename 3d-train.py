import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import gc
import cv2
import os
from os import listdir
import random
from glob import glob
from scipy import ndimage
from sklearn.model_selection import train_test_split

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import kornia
import kornia.augmentation as augmentation
from tqdm import tqdm


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 3e-5
N_EPOCHS = 50
PATIENCE = 10
EXPERIMENTAL = False
AUGMENTATIONS = True

# Config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load metadata
train_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train.csv")
train_bbox = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv")
test_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/test.csv")
ss = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv")

# # Print dataframe shapes
# print('train shape:', train_df.shape)
# print('train bbox shape:', train_bbox.shape)
# print('test shape:', test_df.shape)
# print('ss shape:', ss.shape)
# print('')


# https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344862
bad_scans = ['1.2.826.0.1.3680043.20574','1.2.826.0.1.3680043.29952']

for uid in bad_scans:
    train_df.drop(train_df[train_df['StudyInstanceUID']==uid].index, axis=0, inplace=True)

debug = False
if len(ss) == 3:
    debug = True

    # Fix mismatch with test_images folder
    test_df = pd.DataFrame(columns=['row_id', 'StudyInstanceUID', 'prediction_type'])
    for i in ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876']:
        for j in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall']:
            test_df = test_df._append({'row_id': i + '_' + j, 'StudyInstanceUID': i, 'prediction_type': j},
                                     ignore_index=True)

    # Sample submission
    ss = pd.DataFrame(test_df['row_id'])
    ss['fractured'] = 0.5

# Data augmentations (https://kornia.readthedocs.io/en/latest/augmentation.module.html#geometric)
if AUGMENTATIONS:
    augs = transforms.Compose([
        augmentation.RandomRotation3D((0,0,30), resample='bilinear', p=0.2, same_on_batch=False, keepdim=True),
        augmentation.RandomHorizontalFlip3D(same_on_batch=False, p=0.2, keepdim=True),
        ])
else:
    augs=None


# Dataset for train/valid sets only
class RSNADataset(Dataset):
    # Initialise
    def __init__(self, subset='train', df_table=train_df, transform=None):
        super().__init__()

        self.subset = subset
        self.df_table = df_table.reset_index(drop=True)
        self.transform = transform
        self.targets = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall']

        # Identify files in each of the two datasets
        fh_paths = glob(os.path.join('../input/final_first/train_volumes', "*.pt"))
        sh_paths = glob(os.path.join('../input/final_second/train_volumes', "*.pt"))

        fh_list = []
        sh_list = []
        for i in fh_paths:
            fh_list.append(i.split('\\')[-1][:-3])

        for i in sh_paths:
            sh_list.append(i.split('\\')[-1][:-3])

        self.df_table_fh = self.df_table[self.df_table['StudyInstanceUID'].isin(fh_list)]
        self.df_table_sh = self.df_table[self.df_table['StudyInstanceUID'].isin(sh_list)]

        # Image paths
        self.volume_dir1 = '../input/final_first/train_volumes'  # <=1000 patient
        self.volume_dir2 = '../input/final_second/train_volumes'  # >1000 patient

        # Populate labels
        self.labels = self.df_table[self.targets].values

    # Get item in position given by index
    def __getitem__(self, index):
        if index in self.df_table_fh.index:
            patient = self.df_table_fh[self.df_table_fh.index == index]['StudyInstanceUID'].iloc[0]
            path = os.path.join(self.volume_dir1, f"{patient}.pt")
            vol = torch.load(path).to(torch.float32)
        else:
            patient = self.df_table_sh[self.df_table_sh.index == index]['StudyInstanceUID'].iloc[0]
            path = os.path.join(self.volume_dir2, f"{patient}.pt")
            vol = torch.load(path).to(torch.float32)

        # Data augmentations
        if self.transform:
            vol = self.transform(vol)

        return vol.unsqueeze(0), self.labels[index]

    # Length of dataset
    def __len__(self):
        return len(self.df_table['StudyInstanceUID'])

# Train/valid datasets
experimental=EXPERIMENTAL
if experimental:
    train_table, valid_table = train_test_split(train_df, train_size=0.1, test_size=0.01, random_state=0)
    train_dataset = RSNADataset(subset='train', df_table=train_table, transform=augs)
    valid_dataset = RSNADataset(subset='valid', df_table=valid_table)
else:
    train_table, valid_table = train_test_split(train_df, train_size=0.85, test_size=0.15, random_state=0)
    train_dataset = RSNADataset(subset='train', df_table=train_table, transform=augs)
    valid_dataset = RSNADataset(subset='valid', df_table=valid_table)




# 3D convolutional neural network
class Conv3DNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.norm1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.norm3 = nn.BatchNorm3d(num_features=128)
        self.avg = nn.AdaptiveAvgPool3d((7, 1, 1))
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=896, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=8)

    def forward(self, x):
        # Conv block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm1(out)

        # Conv block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm2(out)

        # Conv block 3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm3(out)

        # Average & flatten
        out = self.avg(out)
        out = self.flat(out)

        # Fully connected layer
        out = self.lin1(out)
        out = self.relu(out)

        # Output layer (no sigmoid needed)
        out = self.lin2(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 3D convolutional neural network
class Conv3DNet_2(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv11 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.norm1 = nn.BatchNorm3d(num_features=16)
        self.inplanes = 16
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm3 = nn.BatchNorm3d(num_features=64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.norm4 = nn.BatchNorm3d(num_features=128)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.norm5 = nn.BatchNorm3d(num_features=256)
        self.avg = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=1024, out_features=512)
        self.lin2 = nn.Linear(in_features=512, out_features=8)

    def forward(self, x):
        # Conv block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv11(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm1(out)

        # CBAM
        out = self.ca(out) * out
        out = self.sa(out) * out

        # Conv block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm2(out)

        # Conv block 3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm3(out)

        # Conv block 4
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm4(out)

        # Conv block 5
        out = self.conv5(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm5(out)

        # Average & flatten
        out = self.avg(out)
        out = self.flat(out)

        # Fully connected layer
        out = self.lin1(out)
        out = self.relu(out)

        # Output layer (no sigmoid needed)
        out = self.lin2(out)

        return out



def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()
# y_hat.shape = (batch_size, num_classes)
# y.shape = (batch_size, num_classes)

# with row-wise weights normalization (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344565)
def competiton_loss_row_norm(y_hat, y):
    loss = loss_fn(y_hat, y.to(y_hat.dtype))
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1)
    w_sum = weights.sum(axis=1)
    loss = torch.div(loss, w_sum)
    return loss.mean()


# model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=8).to(device)
#
model = Conv3DNet_2().to(device)

# # Load checkpoint
# PATH='../output/ckpt-3d/Conv3D_2_loss0.5313.pth'
# if torch.cuda.is_available():
#     checkpoint = torch.load(PATH)
# else:
#     checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
#
# # Load states
# model.load_state_dict(checkpoint['model_state_dict'])

# # Load checkpoint
# PATH='../output/ckpt-3d/Conv3DNet_ori.pt'
# if torch.cuda.is_available():
#     checkpoint = torch.load(PATH)
# else:
#     checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
#
# # Load states
# model.load_state_dict(checkpoint['model_state_dict'])


# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

competition_weights = {
    '-': torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
    '+': torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
}


# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Adam optimiser
optimiser = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(N_EPOCHS/4))

loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = 0.58

# Loop over epochs
for epoch in range(N_EPOCHS):
    model = model.to('cuda')

    # gc_collect()
    loss_acc = 0
    val_loss_acc = 0
    train_count = 0
    valid_count = 0

    # Loop over batches
    for imgs, labels in tqdm(train_loader):
        # Send to device
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        preds = model(imgs)
        L = competiton_loss_row_norm(preds, labels)

        # Backprop
        L.backward()

        # Update parameters
        optimiser.step()

        # Zero gradients
        optimiser.zero_grad()

        # Track loss
        loss_acc += L.detach().item()
        train_count += 1

    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        for val_imgs, val_labels in tqdm(valid_loader):
            # Reshape
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            val_preds = model(val_imgs)
            val_L = competiton_loss_row_norm(val_preds, val_labels)

            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1

    # Save loss history
    loss_hist.append(loss_acc / train_count)
    val_loss_hist.append(val_loss_acc / valid_count)

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {val_loss_acc / valid_count:.5f}')

    # Save model (& early stopping)
    if (val_loss_acc / valid_count) < best_val_loss:
        best_val_loss = val_loss_acc / valid_count
        patience_counter = 0
        print('Valid loss improved --> saving model')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_acc / train_count,
            'val_loss': val_loss_acc / valid_count,
        }, f"../output/ckpt-3d-final/Conv3D_2_crop_loss{best_val_loss:.4f}.pth")

        traced = torch.jit.trace(model.cpu(), torch.rand(1, 1, 224, 224, 224))
        traced.save(f'../output/ckpt-3d-final/Conv3D_2_crop_jit_loss{best_val_loss:.4f}.pth')
    else:
        patience_counter += 1

        if patience_counter == PATIENCE:
            break

print('')
print('Training complete!')

# Plot loss
plt.figure(figsize=(10,5))
plt.plot(loss_hist, c='C0', label='loss')
plt.plot(val_loss_hist, c='C1', label='val_loss')
plt.title('Competition metric')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()