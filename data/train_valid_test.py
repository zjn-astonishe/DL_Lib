import os
import shutil
import collections
import math

"""
用于未按文件夹类别分类，带有csv的数据集
"""

#@save
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

#@save
def copyfile(file_name, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(file_name, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 获得训练数据集中样本最少的类别中的样本书
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n*valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

#@save
def reorg_train(data_dir, labels):
    """单独整理训练集，配合read_split_train_valid_data使用"""
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))

#@save
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio, file_name):
    labels = read_csv_labels(os.path.join(data_dir, file_name))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

if __name__== "__main__" :
    data_dir = '/root/dataset/kaggle_dog_tiny/'
    file_name = 'labels.csv'
    labels = read_csv_labels(os.path.join(data_dir, file_name))
    print('# 训练样本: ', len(labels))
    print('# 类别: ', len(set(labels.values())))
    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, valid_ratio, file_name)
