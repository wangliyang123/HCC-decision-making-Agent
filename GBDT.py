import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import shap

###############################################################################
# 1. 读取训练/验证数据（示例）
###############################################################################
# 假设三分类标签列名为 'Tumor Grade'，其值 ∈ {0, 1, 2}
df = pd.read_excel("./selected_features_val.xlsx")

# 特征和三分类标签
X = df.drop(['Name', 'Tumor Grade'], axis=1)
y = df['Tumor Grade']  # 取值 ∈ {0, 1, 2}

###############################################################################
# 2. 划分训练集和验证集
###############################################################################
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y  # 确保三类分布大体一致
)

###############################################################################
# 3. 数据标准化
###############################################################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

###############################################################################
# 4. 训练模型（以GBDT为例）
###############################################################################
clf = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
clf.fit(X_train, y_train)

###############################################################################
# 5. 在验证集上评估
###############################################################################
# 5.1 预测结果
y_pred_val = clf.predict(X_val)
# predict_proba 返回 [n_samples, n_classes]，对应三分类的概率
y_pred_val_prob = clf.predict_proba(X_val)

# 5.2 准确率、精确率、召回率、F1
accuracy_val = accuracy_score(y_val, y_pred_val)
report_val = classification_report(y_val, y_pred_val, output_dict=True)

# 在三分类下，推荐使用 macro avg（或 weighted avg）衡量整体指标
precision_val = report_val['macro avg']['precision']
recall_val = report_val['macro avg']['recall']
f1_score_val = report_val['macro avg']['f1-score']

print("=== Validation Set Metrics ===")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Macro Precision: {precision_val:.4f}")
print(f"Macro Recall: {recall_val:.4f}")
print(f"Macro F1-Score: {f1_score_val:.4f}")

# 5.3 混淆矩阵
classes = [0, 1, 2]  # 三个类别
conf_matrix_val = confusion_matrix(y_val, y_pred_val, labels=classes)
conf_matrix_val_normalized = conf_matrix_val.astype('float') / conf_matrix_val.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={"size": 16})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix ', fontsize=16)
plt.savefig('./result/Confusion_Matrix_GBDT_val.png', dpi=1000)
plt.close()

# 5.4 ROC AUC（One-vs-Rest + micro-average）
# 将 y_val 转为二进制阵列，用于多分类 ROC
y_val_bin = label_binarize(y_val, classes=classes)  # shape=[n_samples, n_classes]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i, c in enumerate(classes):
    fpr[c], tpr[c], _ = roc_curve(y_val_bin[:, i], y_pred_val_prob[:, i])
    roc_auc[c] = auc(fpr[c], tpr[c])

# micro-average
fpr['micro'], tpr['micro'], _ = roc_curve(y_val_bin.ravel(), y_pred_val_prob.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, c in enumerate(classes):
    plt.plot(fpr[c], tpr[c], color=colors[i], lw=2,
             label=f'Class {c} (AUC={roc_auc[c]:.4f})')
# micro
plt.plot(fpr['micro'], tpr['micro'], color='black', lw=2, linestyle='--',
         label=f'Micro-average (AUC={roc_auc["micro"]:.4f})')

print(f'''Macro AUC: {roc_auc["micro"]:.4f}''')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves ')
plt.legend(loc='lower right')
plt.savefig('./result/ROC_Curve_GBDT_val.png', dpi=1000)
plt.close()

# 5.5 校准曲线（对三分类需要分类别做 One-vs-Rest）
plt.figure(figsize=(8, 6))
for i, c in enumerate(classes):
    # 将“是否为类 c”当作正例
    y_true_c = (y_val == c).astype(int)
    prob_true_c, prob_pred_c = calibration_curve(y_true_c, y_pred_val_prob[:, i], n_bins=10)
    plt.plot(prob_pred_c, prob_true_c, marker='o', label=f'Class {c}')

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curves ')
plt.legend(loc='upper left')
plt.savefig('./result/Calibration_Curve_GBDT_val.png', dpi=1000)
plt.close()

###############################################################################
# 6. SHAP 可解释性分析（多分类）
###############################################################################
# 6.1 定义一个预测函数：输入 numpy 数组返回各类概率
def model_predict_proba(data):
    """
    data: np.array 或 pd.DataFrame, shape = [n_samples, n_features]
    return: np.array, shape = [n_samples, n_classes]
    """
    return clf.predict_proba(data)

# 6.2 准备背景数据（Background Data）
#    - 为了减少计算量，一般从训练集中随机抽取一部分作为背景数据
#    - 如果 X_train 是标准化后的 np.array，需要把它先转为 DataFrame（并注意列名）
#      也可以直接用未标准化的原始特征作参考背景，视情况而定。
X_train_df = pd.DataFrame(X_train, columns=X.columns)

# 随机选取 100 条作为背景数据（数量可根据实际需求调整）
X_background = X_train_df.sample(n=100, random_state=42)

# 6.3 对验证集做 SHAP
X_val_df = pd.DataFrame(X_val, columns=X.columns)  # 保留特征名，便于可视化

# 初始化 KernelExplainer（以对数几率 link='logit' 为例，若不需要可省略参数）
explainer = shap.KernelExplainer(
    model_predict_proba,
    data=X_background,  # 背景数据
    link="logit"
)

# 这里示例对验证集所有样本做解释，若验证集过大也可只选部分
shap_values_val = explainer.shap_values(X_val_df)

# shap_values_val 是一个 list，长度=3，对应三分类
# 6.4 可视化：以“类别0”为例
plt.figure(figsize=(10, 12))

shap.summary_plot(
    shap_values_val[:,:,0],       # 取第 0 类别的 SHAP 值
    X_val_df, 
    feature_names=X.columns,  # 指定特征名
    show=False, 
    plot_size=(15, 12)
)
plt.savefig('./result/SHAP_Summary_GBDT_val.png', dpi=1000)
plt.close()

###############################################################################
# 7. 加载外部测试数据并测试（与训练/验证类似）
###############################################################################
# 假设外部测试集也有三分类标签，与训练集特征结构一致
df_test = pd.read_excel("./selected_features_test.xlsx")
X_test = df_test.drop(['Name', 'Tumor Grade'], axis=1)
y_test = df_test['Tumor Grade']  # ∈ {0,1,2}

# 标准化
X_test = scaler.transform(X_test)

# 预测
y_pred_test = clf.predict(X_test)
y_pred_test_prob = clf.predict_proba(X_test)

# 评估
accuracy_test = accuracy_score(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test, output_dict=True)
precision_test = report_test['macro avg']['precision']
recall_test = report_test['macro avg']['recall']
f1_test = report_test['macro avg']['f1-score']

print("\n=== Test Set Metrics ===")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Macro Precision: {precision_test:.4f}")
print(f"Macro Recall: {recall_test:.4f}")
print(f"Macro F1-Score: {f1_test:.4f}")

# 混淆矩阵 (3×3)
conf_matrix_test = confusion_matrix(y_test, y_pred_test, labels=classes)
conf_matrix_test_normalized = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={"size": 16})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix ', fontsize=16)
plt.savefig('./result/Confusion_Matrix_GBDT_test.png', dpi=1000)
plt.close()

# 多分类 ROC AUC
y_test_bin = label_binarize(y_test, classes=classes)

fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

for i, c in enumerate(classes):
    fpr_test[c], tpr_test[c], _ = roc_curve(y_test_bin[:, i], y_pred_test_prob[:, i])
    roc_auc_test[c] = auc(fpr_test[c], tpr_test[c])

fpr_test['micro'], tpr_test['micro'], _ = roc_curve(y_test_bin.ravel(), y_pred_test_prob.ravel())
roc_auc_test['micro'] = auc(fpr_test['micro'], tpr_test['micro'])

plt.figure(figsize=(8, 6))
for i, c in enumerate(classes):
    plt.plot(fpr_test[c], tpr_test[c], color=colors[i], lw=2,
             label=f'Class {c} (AUC={roc_auc_test[c]:.4f})')
# micro
plt.plot(fpr_test['micro'], tpr_test['micro'], color='black', lw=2, linestyle='--',
         label=f'Micro-average (AUC={roc_auc_test["micro"]:.4f})')

print(f'''Macro AUC: {roc_auc_test["micro"]:.4f}''')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves ')
plt.legend(loc='lower right')
plt.savefig('./result/ROC_Curve_GBDT_test.png', dpi=1000)
plt.close()


# 校准曲线 (3-class)
plt.figure(figsize=(8, 6))
for i, c in enumerate(classes):
    y_true_c_test = (y_test == c).astype(int)
    prob_true_c_test, prob_pred_c_test = calibration_curve(y_true_c_test, y_pred_test_prob[:, i], n_bins=10)
    plt.plot(prob_pred_c_test, prob_true_c_test, marker='o', label=f'Class {c}')

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curves ')
plt.legend(loc='upper left')
plt.savefig('./result/Calibration_Curve_GBDT_test.png', dpi=1000)
plt.close()

# SHAP (Test)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
shap_values_test = explainer.shap_values(X_test_df)

# 显示“类别0”的 summary plot
plt.figure(figsize=(10, 12))
shap.summary_plot(
    shap_values_test[:,:,0],
    X_test_df,
    feature_names=X.columns,
    show=False,
    plot_size=(15, 12)
)
plt.savefig('./result/SHAP_Summary_GBDT_test.png', dpi=1000)
plt.close()
