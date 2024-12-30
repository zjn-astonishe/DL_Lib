import os
import shutil
import random
from tqdm import tqdm

def split_dataset(src_dir, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "训练集、验证集和测试集的比例总和必须为1"
    
    # 创建目标文件夹：train, val, test
    for split in ['train2', 'val2', 'test2']:
        split_dir = split
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    # 遍历每个类别文件夹
    categories = [folder for folder in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, folder))]
    
    # 进度条
    for category in tqdm(categories, desc="分割数据集", ncols=100):
        category_dir = os.path.join(src_dir, category)
        # 获取当前类别下的所有图片文件
        image_files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]
        
        # 打乱图片文件列表
        random.shuffle(image_files)
        
        # 计算各个数据集的大小
        total_images = len(image_files)
        train_size = int(train_ratio * total_images)
        val_size = int(val_ratio * total_images)
        test_size = total_images - train_size - val_size
        
        # 切分图片列表
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size+val_size]
        test_files = image_files[train_size+val_size:]
        
        # 创建类别的子文件夹
        for split, files in zip(['train2', 'val2', 'test2'], [train_files, val_files, test_files]):
            split_category_dir = os.path.join(split, category)
            if not os.path.exists(split_category_dir):
                os.makedirs(split_category_dir)
                
            # 将图片文件复制到对应目录
            for file in files:
                src_file = os.path.join(category_dir, file)
                dst_file = os.path.join(split_category_dir, file)
                shutil.copy(src_file, dst_file)

    print("数据集分割完成！")

# 使用示例
src_directory = '/root/autodl-tmp/imagenet100'  # 数据集根目录
split_dataset(src_directory)
