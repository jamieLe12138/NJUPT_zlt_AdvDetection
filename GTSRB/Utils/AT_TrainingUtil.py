import torch
import torch.nn as nn
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent,BasicIterativeMethod
from Model import Target_model
import torch.nn.functional as F
from Utils.MISC import *
import numpy as np
from Utils.AE_Util import Adversarial_Examples_Generator
from torch.utils.data import TensorDataset, DataLoader 
def train_GTSRB_at_model(root,save_dir,
                              selected_classes,
                              pretrained_model_path=None,
                              save=True,
                              num_epochs=10,
                              batch_size=128,
                              model='resnet18',
                              target_model_name='resnet18',
                              target_model_dir="F:\ModelAndDataset\model\GTSRB",
                              criterion = F.cross_entropy,
                              Attack_method=FastGradientMethod,
                              train_eps=0.1,
                              test_eps=0.1,
                              device="cuda"):
    # 加载数据集
    train_class_mapping,train_class_name_mapping,train_Loader=loadData_selected_labels(root=root,
                                                                                       selected_classes=selected_classes,
                                                                                       batch_size=batch_size,
                                                                                       train=True)
    test_class_mapping,test_class_name_mapping,test_Loader =loadData_selected_labels(root=root,
                                                                                     selected_classes=selected_classes,
                                                                                     batch_size=batch_size,
                                                                                     train=False
                                                                                            )
    print(train_class_name_mapping)
    save_path=save_dir+"/GTSRB_{}_{}_at.pth".format(model,len(selected_classes))
    if os.path.exists(save_path) and pretrained_model_path==None:
        print("File {} already existed ,skip trainning!".format(save_path))
        model.load_state_dict(torch.load(pretrained_model_path))
    else:
        # 加载模型
        if model=='resnet18':
            model=Target_model.ResNet18(num_classes=len(selected_classes)).to(device)
        elif model=='vgg19':
            model=Target_model.VGG_19(num_classes=len(selected_classes)).to(device)
        elif model=='densenet169':
            model=Target_model.Densenet169(num_classes=len(selected_classes)).to(device)
        elif model=='mobilenet':
            model=Target_model.MobileNet(num_classes=len(selected_classes)).to(device)

        if pretrained_model_path:
            print("File {} existed , load saved state dict!")
            model.load_state_dict(torch.load(pretrained_model_path))
    #定义分类器
    classifier=PyTorchClassifier(model=model,loss=criterion)
    #定义使用对抗训练的攻击方法
    attacker=Attack_method(estimator=classifier,eps=train_eps)
    #对抗训练
    adv_trainer=AdversarialTrainer(classifier=classifier,attacks=attacker,ratio=0.5)
    for i,(images,labels) in enumerate(train_Loader):
        images=images.to(device)
        images=images.numpy()
        labels=mapping_labels(train_class_mapping,labels)
        labels=labels.unsqueeze(1).long()
        labels=labels.numpy()
        adv_trainer.fit(x=images,y=labels,batch_size=batch_size,nb_epochs=num_epochs)
        # 测试对抗训练模型的正常精度
        # 评估模型
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_Loader:
            images=images.to(device)
            labels=mapping_labels(test_class_mapping,labels)
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy on normal data: {accuracy:.2f}%,{correct}')
    
    #==========针对相应普通模型生成对抗样本===========
    clip_values = (0.0, 1.0)
    # 定义模型
    if target_model_name=='resnet18':
        target_model=Target_model.ResNet18(len(selected_classes))
    elif target_model_name=='vgg19':
        target_model=Target_model.VGG_19(len(selected_classes))
    elif target_model_name=='densenet169':
        target_model=Target_model.Densenet169(len(selected_classes))
    elif target_model_name=='mobilenet':
        target_model=Target_model.MobileNet(len(selected_classes))
    # 定义优化器
    if target_model_name=='resnet18' or target_model_name=='densenet169':
        optimizer='Adam'
    elif target_model_name=='vgg19' : 
        optimizer="SGD"
    elif target_model_name=='mobilenet':
        optimizer="RMSprop"
    target_model.load_state_dict(torch.load(join(target_model_dir,"GTSRB_{}_{}.pth".format(target_model_name,len(selected_classes)))))
    target_classifier=PyTorchClassifier(model=target_model,loss=nn.CrossEntropyLoss(),
                                optimizer=optimizer,
                                input_shape=(3,64,64), nb_classes=len(selected_classes),clip_values=clip_values)
    attacker=Attack_method(estimator=target_classifier,eps=test_eps)
    # 对抗样本生成器
    ae_generator=Adversarial_Examples_Generator(
            targetmodel=target_model,
            method=attacker,
            targeted=False,
            batch_size=64,
            save_dir=None,
            device=device
            )
    
    for images, labels in test_Loader:
        images=images.to(device)
        labels=mapping_labels(test_class_mapping,labels)
        
        print("Generating AEs!")
        # 生成对抗样本
        test_class_mapping,test_class_name_mapping,test_Loader =loadData_selected_labels(root=root,
                                                                                     selected_classes=selected_classes,
                                                                                     batch_size=batch_size,
                                                                                     train=False
                                                                                    )
        raw_imgs,adv_imgs,raw_labels,adv_labels=ae_generator.generate(test_Loader,test_class_mapping)
        rawDataset=TensorDataset(raw_imgs,raw_labels)
        advDataset=TensorDataset(adv_imgs,adv_labels)
        rawLoader=DataLoader(dataset=rawDataset,batch_size=batch_size,shuffle=False)
        advLoader=DataLoader(dataset=advDataset,batch_size=batch_size,shuffle=False)
        outputs = model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on adversarial data: {accuracy:.2f}%,{correct}')

    
    
    
    

    
