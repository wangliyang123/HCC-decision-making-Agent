README for HCC-decision-making-Agent

This project presents an integrated framework combining radiomics, deep learning (via a modified GhostNet architecture), and LLM-based clinical decision agents for hepatocellular carcinoma (HCC). The following scripts and files are included in the repository, along with guidelines for setting up the environment, training the models, and performing inference.

File Descriptions
GBDT.py: Implements the Gradient Boosting Decision Tree model for classification tasks.
LGBM.py: Implements the LightGBM model, optimized for speed and performance.
RF.py: Implements the Random Forest algorithm used for ensemble learning.
XGB.py: Implements the XGBoost model, a scalable and efficient gradient boosting method.
lasso.py: Performs feature selection using Lasso regression.
model weight.txt: Provides a Baidu Netdisk link to download the pre-trained model weights due to size limitations of GitHub.
preprocessing.py: Contains scripts for image preprocessing including resizing, normalization, and augmentation.
radiomics feature.py: Extracts and processes radiomics features from the segmented MRI tumor regions.
test_ghostnet_rca_cbam_dila_cs.py: Tests the modified GhostNet model with RCA, CBAM, and dilated convolutions.
train_densenet_mlp_cat.py: Trains a DenseNet model integrated with an MLP classifier.
train_fbnet_mlp_cat.py: Trains an FBNet model integrated with an MLP classifier.
train_ghostnet_mlp_cat.py: Trains the standard GhostNet model with an MLP classifier.
train_ghostnet_rca_cbam_dila_cs.py: Trains the enhanced GhostNet model with RCA, CBAM, and dilated convolutions.
train_resnet50_mlp_cat.py: Trains a ResNet50 model with MLP for final classification.
val_densenet_mlp_cat.py: Validates the DenseNet+MLP model on validation datasets.
val_fbnet_mlp_cat.py: Validates the FBNet+MLP model on validation datasets.
val_ghostnet_mlp_cat.py: Validates the standard GhostNet+MLP model.
val_ghostnet_rca_cbam_dila_cs.py: Validates the enhanced GhostNet model with all architectural upgrades.
val_resnet50_mlp_cat.py: Validates the ResNet50+MLP model.
Environment Setup

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- scikit-learn
- pandas
- numpy
- SimpleITK
- pyradiomics
- timm
Install dependencies using:
pip install -r requirements.txt

Data Format

- MRI images are organized into patient folders.
- Each patient folder contains preprocessed image patches.
- Radiomics features are stored in an Excel sheet `selected_features_val.xlsx`.
- Labels are binary (0/1) indicating presence or absence of pathological markers.

Model Training

- Use `train_*.py` scripts to train different architectures.
- Run `train_ghostnet_rca_cbam_dila_cs.py` to train the improved GhostNet model.
- Training output includes logs and `.pth` model weights (referenced via Baidu Netdisk).

Model Evaluation

- Use `val_*.py` scripts for validation after training.
- Evaluation metrics include accuracy, precision, recall, and F1-score.

Model Weights Download

Due to GitHub size limits, trained model weights are available via Baidu Netdisk.
Link: https://pan.baidu.com/s/1UJpKQnzYGCQIZIs5BOQr_w
Password: gjrm

Contact

For data access requests (post-publication) or additional support, please contact the corresponding author: Dr. Liyang Wang.
