import hashlib
import os
import tarfile
import zipfile
import requests
import torch
import torchvision
from torchvision import transforms
import pandas as pd

#@save
DATA_HUB = dict()
DATA_URL = ""

def set_data_hub(DATA_URL, dataset_name, file_name, sha1):
    DATA_HUB[dataset_name] = (DATA_URL + file_name, sha1)
    return DATA_HUB


def download_FashionMNIST(path="/home/zjn/Documents/DataSet"):
    trans = transforms.ToTensor
    mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, transform=trans, download=True)
    return mnist_train, mnist_test

def download_excel(name, cache_dir='/home/zjn/Documents/DataSet'):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None): #@save
    """下载并解压zip/tar文件"""
    fname = download_excel(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'

    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download_excel(name)

if __name__== "__main__" :
    # DATA_HUB = set_data_hub('http://d2l-data.s3-accelerate.amazonaws.com/', 'kaggle_house_train', 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    # train_data = pd.read_csv(download_excel('kaggle_house_train'))
    # print(train_data)
    DATA_HUB = set_data_hub('http://d2l-data.s3-accelerate.amazonaws.com/', 'cifar10_tiny', 'kaggle_cifar10_tiny.zip', '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
    demo = True
    if demo:
        data_dir = download_extract('cifar10_tiny')
    else:
        data_dir = '/home/zjn/Documents/DataSet/cifar-10/'