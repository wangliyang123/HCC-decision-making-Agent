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

class FusionModel(nn.Module):
    def __init__(self, clf_model1,clf_model2):
        super(FusionModel, self).__init__()
        self.clf_model1 = clf_model1
        self.clf_model2 = clf_model2
        self.clf_model1.fc = nn.Identity()
        # self.clf_model2.fc = nn.Identity()
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
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

def evaluate_model(model, device, dataloader, criterion):
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
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * patches.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    average_loss = loss / len(dataloader.dataset)
    accuracy = correct / total
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
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
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
        
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/resnet50_mlp_cat/result/ROC_Curve_val.png', dpi=1000)

if __name__ == "__main__":
    # File paths for data
    val_data_fp = '/mnt/newdisk/KJY/RW39/data/data/内部_Cancer_Embolus_train_val/val'

    # Load test data
    val_loader = data_process(val_data_fp)

    # Load the best model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    clf_model1 = models.resnet50(pretrained=True)
    clf_model2 = nn.Sequential(
            nn.Linear(23, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
    )
    
    fusion_model = FusionModel(clf_model1,clf_model2)
    fusion_model.load_state_dict(torch.load('/mnt/newdisk/KJY/多模态/Cancer_Embolus/resnet50_mlp_cat/model/best_model.pth'))
    fusion_model=fusion_model.to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on the test set
    test_loss, test_accuracy, test_recall, test_precision, test_f1, cm_display, all_labels, all_probs = evaluate_model(fusion_model, device, val_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1-score: {test_f1:.4f}")

    # Plot Confusion Matrix
    cm_display.plot(cmap='Blues',text_kw={'fontsize': 14},values_format=".2f")
    plt.savefig('/mnt/newdisk/KJY/多模态/Cancer_Embolus/resnet50_mlp_cat/result/Confusion_Matrix_val.png', dpi=1000)
    
    # Plot ROC Curve
    plot_roc(all_labels, all_probs, num_classes=2)
