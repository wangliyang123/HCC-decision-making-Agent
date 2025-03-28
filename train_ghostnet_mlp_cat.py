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


class FusionModel(nn.Module):
    def __init__(self, clf_model1,clf_model2):
        super(FusionModel, self).__init__()
        self.clf_model1 = clf_model1
        self.clf_model2 = clf_model2
        self.clf_model1.classifier = nn.Identity()
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
    if not os.path.exists('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/model'):
        os.makedirs('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/model')
    if not os.path.exists('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/result'):
        os.makedirs('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/result')

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
            torch.save(best_model_weights, '/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/model/best_model1.pth')

    print(f"Best validation accuracy: {best_acc:.4f}")

    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, '#1f77b4', label='Training loss')
    plt.plot(epochs, val_losses, '#ff7f0e', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('./result', exist_ok=True)
    plt.savefig("/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/result/loss.png", dpi=1000)

    plt.figure()
    # plt.plot(epochs, train_accs, '#1f77b4', label='Training Accuracy')
    plt.plot(epochs, val_accs, '#2ca02c', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_mlp_cat/result/accuracy.png", dpi=1000)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clf_model1 = timm.create_model('ghostnet_100', pretrained=True)
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
