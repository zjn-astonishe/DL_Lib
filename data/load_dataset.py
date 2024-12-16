import torch
import torchvision
import os
import json
import random
from deal_data import deal_data
from matplotlib import pyplot as plt

def read_split_train_valid_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    dataset_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    dataset_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(dataset_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # 将类别记录在json
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in dataset_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(dataset_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(dataset_class)), dataset_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_test_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    dataset_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    dataset_class.sort()
    # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(dataset_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # 将类别记录在json
    # with open('class_indices.json', 'w') as json_file:
        # json_file.write(json_str)

    test_images_path = []  # 存储训练集的所有图片路径
    test_images_label = []  # 存储训练集图片对应索引信息
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in dataset_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        # image_class = class_indices[cla]
        # 记录该类别的样本数量
        # every_class_num.append(len(images))
        # 按比例随机采样验证样本
        # val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            test_images_path.append(img_path)
            # if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            #     val_images_path.append(img_path)
            #     val_images_label.append(image_class)
            # else:  # 否则存入训练集
            #     train_images_path.append(img_path)
            #     train_images_label.append(image_class)

    # print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(test_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))
    assert len(test_images_path) > 0, "number of training images must greater than 0."
    # assert len(val_images_path) > 0, "number of validation images must greater than 0."

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(dataset_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(dataset_class)), dataset_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

    return test_images_path



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

#@save 
def load_dataset(data_dir, transform_train, transform_test):
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train) for folder in ['train', 'train_valid']
    ]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder), 
        transform=transform_test) for folder in ['valid', 'test']
    ]
    return train_ds, train_valid_ds, valid_ds, test_ds

def deal_dataset(train_ds, train_valid_ds, valid_ds, test_ds, batch_size):
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)
    ]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
    return train_iter, train_valid_iter, valid_iter, test_iter

if __name__== "__main__" :
    # data_dir = '/home/zjn/Documents/DataSet/kaggle_cifar10_tiny/'
    data_dir = '/root/dataset/cifar10/train'
    demo = True
    batch_size = 32 if demo else 128
    transform_train, transform_test = deal_data()
    train_ds, train_valid_ds, valid_ds, test_ds = load_dataset(data_dir, transform_train, transform_test)
    print(test_ds)
    train_iter, train_valid_iter, valid_iter, test_iter = deal_dataset(train_ds, train_valid_ds, valid_ds, test_ds, batch_size)
    print(train_iter)

    