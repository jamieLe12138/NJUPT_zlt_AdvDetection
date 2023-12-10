import numpy as np

import os
from os.path import join

from torch.utils import data

from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToPILImage

from PIL import Image


import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from time import time

import math
def drawGTSRBImages(imgs,labels,label_name,save_path,img_num=64,num_rows=8,show=False,overwrite=False):
    # matplotlib.use('TkAgg') 
    if os.path.exists(save_path)==False or overwrite:
        num_lines = math.ceil(img_num / num_rows)  # 使用math.ceil确保不为整数时向上取整
        fig, axes = plt.subplots(num_lines, num_rows, figsize=(64 , 64))
        labels_name = ["not_"+label_name, label_name]
        for i in range(img_num):
            image = imgs[i]
            label = labels_name[labels[i].item()]
            img = np.transpose(image, (1, 2, 0))  # 重新排列通道维度为 (H, W, C)

            # 计算子图的行号和列号
            row = i // num_rows
            col = i % num_rows

            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Label: {label}",fontsize=25)
            axes[row, col].axis('off')
        if show:
            plt.show()
        plt.savefig(save_path)
    else:
        print("Image {} exists.".format(save_path))
import torch
def get_one_hot_label(labels,num_classes=2):
    #shape(batch_size,1)
    # 创建一个全零的独热编码标签张量
    num_samples = labels.shape[0]
    num_classes = 2
    one_hot_labels = torch.zeros(num_samples, num_classes)
    # 使用scatter_函数将整数数据映射到独热编码标签
    one_hot_labels.scatter_(1, labels, 1)
    return one_hot_labels
from torch.utils.data import DataLoader, SubsetRandomSampler
def loadData_selected_labels(selected_classes,batch_size,train=True):
    transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小为统一大小
    transforms.ToTensor(),
    ])
    if train:
        # 下载和加载GTSRB数据集
        dataset = datasets.GTSRB(root='F:\ModelAndDataset\data',split="train",transform=transform,download=True)
    else:
        # 下载和加载GTSRB数据集
        dataset = datasets.GTSRB(root='F:\ModelAndDataset\data',split="test",transform=transform,download=True)
    print("===========================================")
    # 数字标签与符号映射字典
    class_name_mapping = {
        0: "20 km/h",1: "30 km/h",2: "50 km/h",3: "60 km/h",4: "70 km/h",
        5: "80 km/h",6: "80 km/h(e)",7: "100 km/h",8: "120 km/h",9: "No overtaking",
        10: "No overtaking(trucks)",11: "Crossroad",12: "Priority at next intersection",
        13: "Give way",14: "Stop",15: "No vehicles",16: "No trucks",
        17: "No entry",18: "General caution",19: "Dangerous curve left",
        20: "Dangerous curve right",21: "Double curve",22: "Bumpy road",
        23: "Slippery road",24: "Road narrows on the right",25: "Road work",
        26: "Traffic signals",27: "Pedestrians",28: "Children crossing",
        29: "Bicycles crossing",30: "Beware of ice/snow",31: "Wild animals crossing",
        32: "Passing limits",33: "Turn right ahead",34: "Turn left ahead",
        35: "Ahead only",36: "Go straight or right",37: "Go straight or left",
        38: "Keep right",39: "Keep left",40: "Roundabout mandatory",
        41: "End of no passing",42: "End of no passing (3.5t)"}
    # 筛选指定类别数据
    subset_indices = [idx for idx, (_, label) in enumerate(dataset) if label in selected_classes]
    sampler = SubsetRandomSampler(subset_indices)
    selected_class_mapping={}
    selected_class_name_mapping={}
    for i in range(len(selected_classes)):
        selected_class_mapping[i]=selected_classes[i]
        selected_class_name_mapping[i]=class_name_mapping[selected_classes[i]]
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    return selected_class_mapping,selected_class_name_mapping,data_loader


