import os
import pandas as pd
from radiomics import featureextractor as FEE

# 文件名
para_name = 'E:/HCC.yaml'

# 文件夹列表
folders = ['F-','G-','H-']

# 使用配置文件初始化特征抽取器
extractor = FEE.RadiomicsFeatureExtractor(para_name)

# 准备一个空的DataFrame来存储特征
all_features = pd.DataFrame()

# 遍历每个文件夹
for folder in folders:
    base_folder = os.path.join("F:/tiqu", folder)
    data_folder = os.path.join(base_folder, 'data')
    label_folder = os.path.join(base_folder, 'label')

    # 获取文件列表
    data_files = os.listdir(data_folder)
    label_files = os.listdir(label_folder)

    # 遍历data和label文件夹
    for index, (data_file, label_file) in enumerate(zip(data_files, label_files)):
        ori_path = os.path.join(data_folder, data_file)
        lab_path = os.path.join(label_folder, label_file)

        # 运行特征提取
        result = extractor.execute(ori_path, lab_path)

        # 将特征数据转换为DataFrame，并添加到所有特征的DataFrame中
        features_df = pd.DataFrame({key: [value] for key, value in result.items()})
        features_df['Name'] = folder + "-" + data_file.split('.')[0]
        all_features = all_features.append(features_df, ignore_index=True)

        # 打印进度
        print(f"Processed {index + 1}/{len(data_files)}: {data_file}")

# 写入Excel
all_features.to_excel("F:/tiqu/radiomics_features.xlsx", index=False)
