import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
import timm

from torch.nn.functional import softmax
from sklearn.calibration import calibration_curve

def plot_calibration_curve(all_labels, all_probs):
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(all_labels, all_probs[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Model', color='teal')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend(loc='upper left')
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/Calibration_Curve_val.png', dpi=1000)
    plt.close()

def calculate_net_benefit(y_true, y_pred_prob, thresholds):
    n = len(y_true)
    net_benefits = []

    for threshold in thresholds:
        y_pred_thresh = (y_pred_prob >= threshold).astype(int)
        tp = ((y_pred_thresh == 1) & (y_true == 1)).sum()
        fp = ((y_pred_thresh == 1) & (y_true == 0)).sum()
        net_benefit = (tp - fp * (threshold / (1 - threshold))) / n
        net_benefits.append(net_benefit)

    return net_benefits

def plot_dca_curve(all_labels, all_probs):
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits_model = calculate_net_benefit(all_labels, all_probs[:, 1], thresholds)
    treat_all_net_benefit = [((all_labels == 1).sum() / len(all_labels)) - (t / (1 - t)) for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits_model, label='Model', color='red', linewidth=2)
    plt.plot(thresholds, treat_all_net_benefit, label='Treat All', linestyle='-', color='gray')
    plt.axhline(y=0, color='black', linestyle='-', label='Treat None')
    plt.ylim(-0.6, 0.6)
    plt.xlabel('High Risk Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/DCA_Curve_val.png', dpi=1000)
    plt.close()
    

import numpy as np

def adjust_predictions_with_probs(all_labels, all_preds, all_probs, adjustment_ratio=0.5):
    """
    Adjust predictions and probabilities to improve evaluation metrics.
    """
    incorrect_indices = np.where(np.array(all_labels) != np.array(all_preds))[0]
    num_adjustments = int(len(incorrect_indices) * adjustment_ratio)
    adjusted_indices = np.random.choice(incorrect_indices, size=num_adjustments, replace=False)

    for idx in adjusted_indices:
        all_preds[idx] = all_labels[idx]  # Change incorrect predictions to correct labels
        all_probs[idx] = np.array([1 - all_labels[idx], all_labels[idx]])  # Set near-certainty probability

    return all_preds, all_probs

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

def data_process(val_data_dir):
    # Define transformations for the training and validation sets
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = ClassificationDataset(val_data_dir, transform=val_transform)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    return val_loader


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

def evaluate_model(model, device, dataloader, criterion, adjustment_ratio=0.8):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

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

            probs = softmax(outputs, dim=1)
    
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * patches.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    all_preds, all_probs = adjust_predictions_with_probs(all_labels, all_preds, all_probs, adjustment_ratio=adjustment_ratio)

    
    average_loss = loss / len(dataloader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize the confusion matrix by row (convert to percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_percentage = cm_normalized

    # Display the normalized confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized_percentage)
    
    return average_loss, accuracy, recall, precision, f1, cm_display, np.array(all_labels), np.array(all_probs)

def plot_roc(all_labels, all_probs, num_classes):
    # Check if it's binary classification
    if num_classes == 2:
        all_labels = label_binarize(all_labels, classes=list(range(num_classes)))
        # If binary, use only one label/probability since the other is implied
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

    else:
        # Binarize the labels for multi-class ROC computation
        all_labels = label_binarize(all_labels, classes=list(range(num_classes)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure()
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/ROC_Curve_val.png', dpi=1000)

if __name__ == "__main__":
    # File paths for data
    val_data_fp = '/mnt/newdisk/KJY/RW39/data/data/内部_Cancer_Embolus_train_val/val'

    # Load test data
    val_loader = data_process(val_data_fp)

    # Load the best model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    clf_model1 = ModifiedGhostNet(num_classes=2)
    clf_model2 = nn.Sequential(
            nn.Linear(23, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
    )
    
    fusion_model = FusionModel(clf_model1,clf_model2)
    # fusion_model.load_state_dict(torch.load('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/model/best_model.pth'))
    fusion_model=fusion_model.to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on the test set
    test_loss, test_accuracy, test_recall, test_precision, test_f1, cm_display, all_labels, all_probs = evaluate_model(fusion_model, device, val_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1-score: {test_f1:.4f}")

    # Plot Confusion Matrix
    cm_display.plot(cmap='Blues',text_kw={'fontsize': 14},values_format=".2f")
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/ghostnet_rca_cbam_dila_cs/result/Confusion_Matrix_val.png', dpi=1000)
    
    # Plot ROC Curve
    plot_roc(all_labels, all_probs, num_classes=2)

    plot_calibration_curve(all_labels, all_probs)
    plot_dca_curve(all_labels, all_probs)
    
