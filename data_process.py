import os
import shutil
import numpy as np
import pandas as pd

img_path = 'fackface_detect_1/深度伪造人脸检测数据集/image/train'
label_path = pd.read_csv('fackface_detect_1/深度伪造人脸检测数据集/train.labels.csv')
target_train_path = 'fakeface/train'
target_test_path = 'fakeface/test'
if not os.path.exists(target_train_path + '/0'):
    os.makedirs(target_train_path + '/0')
if not os.path.exists(target_train_path + '/1'):
    os.makedirs(target_train_path + '/1')
if not os.path.exists(target_test_path + '/0'):
    os.makedirs(target_test_path + '/0')
if not os.path.exists(target_test_path + '/1'):
    os.makedirs(target_test_path + '/1')
label_path = np.array(label_path)
true_num, false_num = 0, 0
for item in label_path:
    img_name, label = item[0][:-2], item[0][-1]
    if label == '0':
        true_num += 1
    else:
        false_num += 1
test_rate = 0.1
true_num_train = 0
false_num_train = 0
for item in label_path:
    img_name, label = item[0][:-2], item[0][-1]
    if label == '0':
        if true_num_train <= int(true_num * (1 - test_rate)):
            true_num_train += 1
            shutil.copy(img_path + '/' + img_name, target_train_path + '/0/' + img_name)
        else:
            shutil.copy(img_path + '/' + img_name, target_test_path + '/0/' + img_name)
    else:
        if false_num_train <= int(false_num * (1 - test_rate)):
            false_num_train += 1
            shutil.copy(img_path + '/' + img_name, target_train_path + '/1/' + img_name)
        else:
            shutil.copy(img_path + '/' + img_name, target_test_path + '/1/' + img_name)
    print(img_name)


