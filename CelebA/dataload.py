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
class CELEBA_GEN(data.Dataset):
    def __init__(self,root,label,model,attr_name,transform):
        self.root =root
        self.label=label
        self.transform=transform
        img_load_path=join(self.root,attr_name+"_"+model+"_"+"image.pth")
        label_load_path=join(self.root,attr_name+"_"+model+"_"+"label.pth")

        print('Load images from:',img_load_path)
        print('Load labels from:',label_load_path)
        self.data=np.load(img_load_path, mmap_mode='r')
        self.data=self.data.transpose((0, 2, 3, 1))
        self.labels=np.load(label_load_path)
    def __getitem__(self, index):
        
        img= self.data[index]
        target= self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray((img* 255).astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)
        target = target.astype(int)
        return img, target
    def __len__(self):
        return len(self.data)
    
class CELEBA_Attack(data.Dataset):
    def __init__(self,root,adv,attackMethod,model,label,transform):
        self.root =root
        self.adv =adv
        self.attackMethod=attackMethod
        self.model=model
        self.label=label
        self.transform=transform
        img_load_path=self.root
        label_load_path=self.root
        self.taskname=self.attackMethod+"_"+self.model+"_"+self.label

        if self.adv:
           img_load_path+="/AdvImage_"+self.taskname+".npy"
           label_load_path+="/AdvLabel_"+self.taskname+".npy"
        else:
           img_load_path+="/RawImage_"+self.taskname+".npy"
           label_load_path+="/RawLabel_"+self.taskname+".npy"
        print('Load images from:',img_load_path)
        print('Load labels from:',label_load_path)
        self.data=np.load(img_load_path, mmap_mode='r')
        self.data=self.data.transpose((0, 2, 3, 1))

        self.labels=np.load(label_load_path)
    def __getitem__(self, index):
        
        img= self.data[index]
        target= self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        # print(type(img))
        # print(target.shape)
        # print(type(target))
        img = Image.fromarray((img* 255).astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)
        target = target.astype(int)

        return img, target

    def __len__(self):
        return len(self.data)
class CELEBA(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``celebA`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """


    def __init__(self, root, train=True,train_ratio=0.7,transform=None, label='Smiling'):
        attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.root = root
        self.train = train  # training set or test set
        self.filename='celebA'
        self.transform=transform
        self.idx = attributes.index(label)
        print(self.idx)
        # now load the picked numpy arrays
        data_all=np.load(join(self.root, self.filename, 'xTrain.npy'), mmap_mode='r')
        data_size=len(data_all)

        if train==True:
            self.size=int(train_ratio*data_size)
            print('Load Train dataset:',self.size)
        else:
            self.size=data_size-int(train_ratio*data_size)
            print('Load Test dataset:',self.size)
        
        label_all=np.load(join(self.root, self.filename, 'yAllTrain.npy'))[:,self.idx]
        if self.train:
            self.train_data = data_all[:int(train_ratio*data_size)]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            train_labels = label_all[:int(train_ratio*data_size)]
            self.train_labels = (train_labels.astype(int)+1) // 2
            # print(np.shape(self.train_labels), np.shape(self.train_data)) 
            # print(np.unique(self.train_labels)) 

        else:
            self.test_data =data_all[int(train_ratio*data_size):]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            test_labels = label_all[int(train_ratio*data_size):]
            self.test_labels = (test_labels.astype(int)+1) // 2
 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target.astype(int)


        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_dir_exist(self):
        inDir=join(self.root, self.filename)
        assert os.path.isdir(inDir)
        assert os.path.exists(join(inDir, 'xTrain.npy'))
        assert os.path.exists(join(inDir, 'yAllTrain.npy'))
import math
def drawCelebAImages(imgs,labels,label_name,save_path,img_num=64,num_rows=8,show=False):
    # matplotlib.use('TkAgg') 
    if os.path.exists(save_path)==False:
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


