import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from scipy.stats import sem, t
from sklearn.utils import resample

# 读取excel文件
df = pd.read_excel('E:/xx.xlsx')

# 划分特征和标签
X = df.iloc[:, 2:]
y = df.iloc[:, 0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进行LassoCV
clf = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_train, y_train)

# 打印并保存最优阿尔法值
optimal_alpha = clf.alpha_
print("最优阿尔法值：", optimal_alpha)

# 保存阿尔法值到新的excel文件
alpha_df = pd.DataFrame({'Optimal Alpha': [optimal_alpha]})
alpha_df.to_excel('E:/xx.xlsx', index=False)

# 绘制LassoCV收敛图
m_log_alphas = -np.log10(clf.alphas_ + np.finfo(float).eps)
plt.figure()
plt.plot(m_log_alphas, clf.mse_path_, ':')
plt.plot(m_log_alphas, clf.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(clf.alpha_ + np.finfo(float).eps), linestyle='--', color='k',
            label='alpha: CV estimate')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent')
plt.axis('tight')
plt.show()

# AUC of each feature
alpha_aucs = []
bootstrap_size = 10

for alpha in clf.alphas_:
    aucs = []
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    # Perform bootstrap resampling
    for _ in range(bootstrap_size):
        while True:
            X_resample, y_resample = resample(X_test, y_test)
            if len(np.unique(y_resample)) > 1:
                break
        y_pred = lasso.predict(X_resample)
        auc = roc_auc_score(y_resample, y_pred)
        aucs.append(auc)

    alpha_aucs.append(aucs)

# Get mean and confidence interval for each alpha
alpha_means = [np.mean(aucs) for aucs in alpha_aucs]
alpha_confidence_low = [np.percentile(aucs, 2.5) for aucs in alpha_aucs]  # 2.5 percentile
alpha_confidence_high = [np.percentile(aucs, 97.5) for aucs in alpha_aucs]  # 97.5 percentile

m_log_alphas = -np.log10(clf.alphas_)

# 绘制不同特征筛选下的AUC图
plt.figure()
plt.plot(m_log_alphas, alpha_means, label="Mean AUC")
plt.fill_between(m_log_alphas, alpha_confidence_low, alpha_confidence_high, color='b', alpha=.1,
                 label="95% Confidence Interval")
plt.xlabel('-log(alpha)')
plt.ylabel('AUC')
plt.title('AUC vs alpha with Confidence Interval')
plt.legend()
plt.show()

# 特征选择并保存到新的excel中
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
selected_features = pd.DataFrame(X_new, columns=[df.columns[i+1] for i in range(len(model.get_support())) if model.get_support()[i]])
selected_features.to_excel('E:/xx.xlsx')



# 获取每个选中特征的系数，并按系数的绝对值从高到低排序
feature_coefficients = clf.coef_[model.get_support()]
features = [df.columns[i+2] for i in range(len(model.get_support())) if model.get_support()[i]]
feature_coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': feature_coefficients})
feature_coefficients_df['Absolute Coefficient'] = feature_coefficients_df['Coefficient'].abs()
feature_coefficients_df = feature_coefficients_df.sort_values(by='Absolute Coefficient', ascending=False).drop(columns=['Absolute Coefficient'])

# 保存系数到新的excel文件
feature_coefficients_df.to_excel('E:/xx.xlsx', index=False)

# 如果特征数大于30，只保留前30个特征
if len(feature_coefficients_df) > 30:
    feature_coefficients_df = feature_coefficients_df.head(30)


# 绘制特征系数的直方图
plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_coefficients_df, palette='viridis')
plt.title('Top Features and Their Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()
