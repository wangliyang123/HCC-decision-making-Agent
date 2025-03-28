import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn.functional as F
import pandas as pd
import timm

import numpy as np
def extract_files(directory):
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue

        files.append(file_path)
    return files

import random
class ClassificationDataset(Dataset):
    def __init__(self, img_dir,num_patches=100, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = []
        self.classes = ['0', '1']
        self.num_patches=num_patches
        self.df=pd.read_excel('../selected_features_val.xlsx')
        self.df=self.df.drop([ 'Cancer Embolus'], axis=1)
        
        for cls in self.classes:
            img_cls_dir = os.path.join(img_dir, cls)
            patient_id = [f for f in os.listdir(img_cls_dir)]
            for img_name in patient_id :

                extract_files_list=extract_files(os.path.join(img_cls_dir, img_name))
                
                # 设置随机数种子
                random.seed(42)
                sampled_fps = random.sample(extract_files_list, min(self.num_patches, len(extract_files_list)))
                if len(sampled_fps) == 0:
                    continue

                self.img_list.append((sampled_fps, cls,patient_id))
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_fps, cls , patient_id = self.img_list[idx]
        patches=[]
        for img_fp in img_fps:
            image = Image.open(img_fp).convert('RGB')
            patches.append(image)
        
        array = self.df.query("Name == @patient_id").drop(columns=["Name"]).values[0].astype(np.float32)
        seqs = [torch.tensor(array,dtype=torch.float32)]*len(img_fps)

        if self.transform:
            patches = [self.transform(patch) for patch in patches]
        
        label = self.classes.index(cls)
    
        return torch.stack(patches), torch.stack(seqs) ,label

def data_process(train_data_dir, val_data_dir):
    # Define transformations for the training and validation sets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ClassificationDataset(train_data_dir, transform=train_transform)
    val_dataset = ClassificationDataset(val_data_dir, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader


class DilatedConv(nn.Module):
    """
    膨胀卷积：使用空洞卷积代替标准的深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=2):
        super(DilatedConv, self).__init__()
        # 使用膨胀卷积
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=2):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 4

        # 替换为膨胀卷积
        self.dw_conv1 = DilatedConv(in_channels, mid_channels, stride=stride, dilation=2)
        self.dw_conv2 = DilatedConv(mid_channels, out_channels, stride=1, dilation=2)
        self.groups = groups

        if stride == 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        out = self.dw_conv1(x)
        out = self.channel_shuffle(out, self.groups)
        out = self.dw_conv2(out)

        residual = self.shortcut(x)
        out = out + residual
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out

class RCAModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(RCAModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        self.ca = ChannelAttention(channels, reduction)
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.ca(out)
        out += residual
        return out

class ModifiedGhostNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ModifiedGhostNet, self).__init__()

        model = timm.create_model('ghostnet_100', pretrained=True)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks_0_7 = model.blocks[:8]
        self.cbam8 = CBAM(160)
        self.block9 = model.blocks[8]
        self.block10 = self._make_stage(160, 160, num_blocks=2, stride=2)
        self.rca11 = RCAModule(160)
        self.block12 = model.blocks[9]
        self.global_pool = model.global_pool
        self.conv_head = model.conv_head
        self.act2 = model.act2
        self.flatten = model.flatten
        self.fc = nn.Sequential(
            nn.Linear(1280, num_classes)
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_stem(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.blocks_0_7(out)
        out = self.cbam8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.rca11(out)
        out = self.block12(out)
        out = self.global_pool(out)
        out = self.conv_head(out)
        out = self.act2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

class FusionModel(nn.Module):
    def __init__(self, clf_model1,clf_model2):
        super(FusionModel, self).__init__()
        self.clf_model1 = clf_model1
        self.clf_model2 = clf_model2
        self.clf_model1.fc = nn.Identity()
        # self.clf_model2.fc = nn.Identity()
        
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512+512, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

    def forward(self, img1, seq):
        features1 = self.fc1(self.clf_model1(img1))
        # features2 = self.fc2(self.clf_model(seq))
        features2 = self.clf_model2(seq)
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        fused_features = torch.cat((features1, features2), dim=1)
        output = self.fc2(fused_features)
        return output

def train_fusion_model(fusion_model, train_loader, val_loader, num_epochs=30):
    if not os.path.exists('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/model'):
        os.makedirs('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/model')
    if not os.path.exists('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result'):
        os.makedirs('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    train_losses = []
    val_losses = []
    val_accs = []
    best_acc = 0.0
    best_model_weights = fusion_model.state_dict()

    for epoch in range(num_epochs):
        fusion_model.train()
        train_loss = 0
        train_corrects = 0
        for patches, seqs, labels in tqdm(train_loader):
            patches, seqs, labels = patches.to(device), seqs.to(device), labels.to(device)
            b, p, c, h, w = patches.size()
            patches = patches.view(b * p, c, h, w)
            b, p, l = seqs.size()
            seqs = seqs.view(b * p, l)
            optimizer.zero_grad()
            outputs = fusion_model(patches, seqs)
            outputs = outputs.view(b, p, -1)  # 恢复到 (batch_size, num_patches, num_classes)
            outputs = outputs.mean(dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * patches.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        train_losses.append(train_loss / len(train_loader))

        val_loss = 0
        val_corrects = 0
        fusion_model.eval()
        with torch.no_grad():
            for patches, seqs, labels in val_loader:
                patches, seqs, labels = patches.to(device), seqs.to(device), labels.to(device)
                b, p, c, h, w = patches.size()
                patches = patches.view(b * p, c, h, w)
                b, p, l = seqs.size()
                seqs = seqs.view(b * p, l)
                outputs = fusion_model(patches, seqs)
                outputs = outputs.view(b, p, -1)
                outputs = outputs.mean(dim=1)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * patches.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_losses.append(val_loss / len(val_loader))
        acc = val_corrects.double().cpu().numpy() / len(val_loader.dataset)
        val_accs.append(acc)

        scheduler.step(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f},  Val Acc: {val_accs[-1]:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            best_model_weights = fusion_model.state_dict().copy()
            torch.save(best_model_weights, '/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/model/best_model.pth')

    print(f"Best validation accuracy: {best_acc:.4f}")

    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('./result', exist_ok=True)
    plt.savefig("/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/loss.png", dpi=1000)

    plt.figure()
    # plt.plot(epochs, train_accs, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'g', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/accuracy.png", dpi=1000)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clf_model1 = ModifiedGhostNet(num_classes=2)
    clf_model2 = nn.Sequential(
            nn.Linear(23, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
    )

    fusion_model = FusionModel(clf_model1,clf_model2).to(device)
    train_data_dir, val_data_dir = '/mnt/newdisk/KJY/RW39/data/data/内部_Cancer_Embolus_train_val/train', '/mnt/newdisk/KJY/RW39/data/data/内部_Cancer_Embolus_train_val/val'
    train_loader, val_loader = data_process(train_data_dir, val_data_dir)
    
    train_fusion_model(fusion_model, train_loader, val_loader, num_epochs=30)
