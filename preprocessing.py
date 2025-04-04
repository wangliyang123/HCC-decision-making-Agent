import SimpleITK as sitk
import os
import random
import numpy as np
import cv2

#裁剪预处理
raw_dataset_path = 'E:/xx'
# 预处理后的数据集的输出路径
fixed_dataset_path = 'E:/new_traindata'
if not os.path.exists(fixed_dataset_path):
    os.mkdir(fixed_dataset_path)
# 创建保存目录
if os.path.exists(fixed_dataset_path):
    os.makedirs(os.path.join(fixed_dataset_path,'data'))
    os.makedirs(os.path.join(fixed_dataset_path,'label'))

upper = 200
lower = -200
for ct_file in os.listdir(os.path.join(raw_dataset_path ,'data')):
    # 读取origin
    ct = sitk.ReadImage(os.path.join(os.path.join(raw_dataset_path ,'data'), ct_file), sitk.sitkInt16)
    # 转换成numpy格式
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(os.path.join(raw_dataset_path ,'label'), ct_file.replace('volume', 'segmentation')),
                            sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    print("裁剪前:{}".format(ct.GetSize(), seg.GetSize()))

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 找到肝脏区域开始和结束的slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())

    new_seg = sitk.GetImageFromArray(seg_array)
    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    print("裁剪后:{}".format(new_ct.GetSize(), new_seg.GetSize()))

    sitk.WriteImage(new_ct, os.path.join(os.path.join(fixed_dataset_path ,'data'), ct_file))
    sitk.WriteImage(new_seg,
                    os.path.join(os.path.join(fixed_dataset_path , 'label'), ct_file.replace('volume', 'segmentation')))

# nii批量转化
data_path = 'E:/new_traindata/data'
label_path = 'E:/new_traindata/label'
count = 0
if not os.path.exists('E:/newtestdata1'):
    os.mkdir('E:/newtestdata1')
    os.makedirs(os.path.join('E:/newtestdata1','origin'))
    os.makedirs(os.path.join('E:/newtestdata1','label'))
for f in os.listdir(data_path):
    origin_path= os.path.join(data_path, f)
    seg_path = os.path.join(label_path,f).replace('volume','segmentation')
    origin_array = sitk.GetArrayFromImage(sitk.ReadImage(origin_path))
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    for i in range(seg_array.shape[0]):
        seg_image = seg_array[i,:,:]
        seg_image = np.rot90(np.transpose(seg_image, (1,0)))
        origin_image = origin_array[i,:,:]
        origin_image = np.rot90(np.transpose(origin_image, (1,0)))
        cv2.imwrite('E:/newtestdata1/label/'+str(count) + '.png', seg_image)
        cv2.imwrite('E:/newtestdata1/origin/'+str(count) + '.jpg', origin_image)
        count += 1

print(count)

#建立训练集，验证集，测试集并建立txt
random.seed(2022)
path_origin = 'E:/newtestdata1/origin'
path_label = 'E:/newtestdata1/label'
files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(path_origin)))
random.shuffle(files)
# 训练集和测试集
rate = int(len(files) * 0.8)
train_txt = open('E:/newtestdata1/train_list.txt','w')
val_txt = open('E:/newtestdata1/val_list.txt','w')
test_txt = open('E:/newtestdata1/test_list.txt','w')
for i,f in enumerate(files):
    image_path = os.path.join(path_origin, f)
    label_name = f.split('.')[0]+ '.png'
    label_path = os.path.join(path_label, label_name)
    if i < rate:
        train_txt.write(image_path + ' ' + label_path+ '\n')
    else:
        if i%2 :
            val_txt.write(image_path + ' ' + label_path+ '\n')
        else:
            test_txt.write(image_path + ' ' + label_path+ '\n')
train_txt.close()
val_txt.close()
test_txt.close()
print('完成')

