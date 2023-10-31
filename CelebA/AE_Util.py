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
import os
class Adversarial_Examples_Generator():
    def __init__(self,targetmodel,task,dataset,method,targeted,batch_size,save_dir,save_raw=False,device='cuda'):
        self.targetmodel=targetmodel
        self.task=task
        self.dataset=dataset
        self.method=method
        self.targeted=targeted
        
        self.batch_size=batch_size
        self.save_dir=save_dir
        self.save_raw=save_raw
        self.device=device
        
        
    def generate(self):
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

        if save_dir:
            adv_img_path=save_dir+"/AdvImage_"+str(type(attacker).__name__)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            adv_label_path=save_dir+"/AdvLabel_"+str(type(attacker).__name__)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"

        if save_raw:
            raw_img_path=save_dir+"/RawImage_"+str(type(attacker).__name__)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
            raw_label_path=save_dir+"/RawLabel_"+str(type(attacker).__name__)+"_"+str(type(model)).split(".")[-1].split("'")[0]+"_"+task+".npy"
        
        if os.path.exists(adv_img_path) and os.path.exists(adv_img_path) and os.path.exists(adv_img_path) and os.path.exists(adv_img_path):
            print("Attack Task has been finished!")
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)       
            for batch_idx, (data, ground_truth) in enumerate(loader):
                # 将数据和目标移到正确的设备
                data, ground_truth = data.to(device), ground_truth.to(device)
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
                    # 选出预测正确，符合地面标签的样本
                    predict_correct =torch.eq(raw_predicted,ground_truth)
                    raw_data=raw_data[predict_correct]
                    adv_data=adv_data[predict_correct]
                    raw_predicted=raw_predicted[predict_correct]
                    adv_predicted=adv_predicted[predict_correct]
                

                    #判断是否成功攻击
                    label_commmon = torch.eq(raw_predicted, adv_predicted).to(device)
                    if batch_size%10==0:
                        print("raw_predicted:{}".format(raw_predicted))
                        print("adv_predicted:{}".format(adv_predicted))
                    #添加成功攻击的样本
                    raw_imgs=torch.cat([raw_imgs,raw_data[~label_commmon]],dim=0)
                    adv_imgs=torch.cat([adv_imgs,adv_data[~label_commmon]],dim=0)
                    raw_labels = torch.cat([raw_labels, raw_predicted[~label_commmon].to(device)], dim=0)
                    adv_labels = torch.cat([adv_labels, adv_predicted[~label_commmon].to(device)], dim=0)
            print("Generate {} Adversarial Examples! ".format(len(adv_labels)))
            print("Total Pictures:{}".format(len(dataset)))
            Acc=100*len(adv_labels)/len(dataset)
            print(f"Attack Accuracy:{Acc:.2f}%")

            np.save(adv_img_path, adv_imgs.detach().cpu().numpy())
            print("Save adv_img to {}".format(adv_img_path))

            np.save(adv_label_path, adv_labels.detach().cpu().numpy())
            print("Save adv_label to {}".format(adv_label_path))
        
            np.save(raw_img_path, np.asarray(raw_imgs.detach().cpu().numpy()))
            print("Save raw_img to {}".format(raw_img_path))

            
            np.save(raw_label_path, raw_labels.detach().cpu().numpy().astype('int'))
            print("Save raw_label to {}".format(raw_label_path))
