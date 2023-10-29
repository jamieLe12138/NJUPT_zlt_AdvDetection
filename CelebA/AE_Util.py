import art
from art.attacks.evasion import FastGradientMethod
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
class Adversarial_Examples_Generator():
    def __init__(self,targetmodel,task,dataset,method,targeted,batch_size,save_dir=None,save_raw=False,device='cuda'):
        self.targetmodel=targetmodel
        self.task=task
        self.dataset=dataset
        self.method=method
        self.targeted=targeted
        
        self.batch_size=batch_size
        self.save_dir=save_dir
        self.save_raw=save_raw
        self.device=device
        
        
    def generate(self,num_samples=None):
        model=self.targetmodel
        task=self.task
        device=self.device
        dataset=self.dataset
        attacker=self.method
        batch_size = self.batch_size # 选择适当的批次大小
        save_dir=self.save_dir
        save_raw=self.save_raw
        
        model.to(device)
        #初始化
        raw_imgs=torch.empty(0).to(device)
        adv_imgs=torch.empty(0).to(device)
        raw_labels = torch.empty(0).to(device)
        adv_labels = torch.empty(0).to(device)
        adv_probs = torch.empty(0).to(device)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)       
        for batch_idx, (data, label) in enumerate(loader):
            # 将数据和目标移到正确的设备
            data, label = data.to(device), label.to(device)
            data=data.detach().cpu().numpy()
            # 生成对抗样本
            adv_data = attacker.generate(data) 
            model.eval()
            with torch.no_grad():
                adv_data=torch.from_numpy(adv_data).to(device)
                adv_output=model(adv_data)
                raw_data=torch.from_numpy(data).to(device)
                raw_output=model(raw_data)
                # 从模型的输出中提取标签 
                _,raw_predicted = raw_output.max(1) #获取样本的原标签
                _, adv_predicted = adv_output.max(1)  # 获取每个样本的预测标签 
                #判断是否成功攻击
                comparison = torch.eq(raw_predicted, adv_predicted).to(device)
                if batch_size%10==0:
                    print("raw_predicted:{}".format(raw_predicted))
                    print("adv_predicted:{}".format(adv_predicted))
                    print("comparison:",comparison)
                #添加成功攻击的样本
                raw_imgs=torch.cat([raw_imgs,raw_data[~comparison]],dim=0)
                adv_imgs=torch.cat([adv_imgs,adv_data[~comparison]],dim=0)
                raw_labels = torch.cat([raw_labels, raw_predicted[~comparison].to(device)], dim=0)
                adv_labels = torch.cat([adv_labels, adv_predicted[~comparison].to(device)], dim=0)
                adv_probs = torch.cat([adv_probs, torch.softmax(adv_output[~comparison], dim=1).to(device)], dim=0)
        Acc=100*len(adv_labels)/len(dataset)
        print(f"Attack Accuracy:{Acc:.2f}%")
        if save_dir:
            adv_img_path=save_dir+"/AdvImage_"+str(type(attacker).__name__)+"_"+str(attacker.eps)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            np.save(adv_img_path, adv_imgs.detach().cpu().numpy())
            print("Save adv_img to {}".format(adv_img_path))

            adv_label_path=save_dir+"/AdvLabel_"+str(type(attacker).__name__)+"_"+str(attacker.eps)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            np.save(adv_label_path, adv_labels.detach().cpu().numpy())
            print("Save adv_label to {}".format(adv_label_path))
        if save_raw:
            raw_img_path=save_dir+"/RawImages_"+str(type(attacker).__name__)+"_"+str(attacker.eps)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            np.save(raw_img_path, raw_imgs.detach().cpu().numpy())
            print("Save raw_img to {}".format(raw_img_path))

            raw_label_path=save_dir+"/RawLabels_"+str(type(attacker).__name__)+"_"+str(attacker.eps)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            np.save(raw_label_path, raw_labels.detach().cpu().numpy())
            print("Save raw_label to {}".format(raw_label_path))

        examplePairs=[]
        if num_samples!=None:       
            #随机选取一定数量的对抗样本
            random_indices=random.sample(range(len(raw_imgs)),num_samples)
            for idx in random_indices:
                raw_img=raw_imgs[idx].squeeze().cpu().numpy()
                adv_img=adv_imgs[idx].squeeze().cpu().numpy()
                raw_label=raw_labels[idx].item()
                adv_label=adv_labels[idx].item()
                adv_prob=adv_probs[idx]
                newAdvSamplePair=ExamplePair(idx,raw_img,adv_img,raw_label,adv_label,adv_prob)
                examplePairs.append(newAdvSamplePair)
        return examplePairs




        

class ExamplePair():
    def __init__(self,id,raw_img,adv_img,raw_label,adv_label,adv_prob):
        self.id=id 
        self.raw_img = raw_img
        self.adv_img = adv_img
        self.raw_label = raw_label
        self.adv_label = adv_label
        self.adv_probs = adv_prob
    def save(self,raw_path,adv_path):
        raw_path=raw_path+"raw_"+str(id)+"_"+str(int(self.raw_label))+"_"+str(int(self.raw_label))+".png"
        adv_path=adv_path+"adv_"+str(id)+"_"+str(int(self.raw_label))+"_"+str(int(self.raw_label))+".png"
        plt.imsave(raw_path,self.raw_img,cmap='gray')
        plt.imsave(adv_path,self.adv_img,cmap='gray')