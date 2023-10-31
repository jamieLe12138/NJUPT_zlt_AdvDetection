from AE_Util import Adversarial_Examples_Generator
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod,BasicIterativeMethod,ProjectedGradientDescent,PixelAttack,ZooAttack,CarliniL2Method
import sys
sys.path.append("E:/Project/ZLTProgram/")
from Target_model import Target_model
import torch
import torch.nn as nn
from dataload import CELEBA,CELEBA_Attack,drawCelebAImages
from torchvision import transforms
from TargetModelTrainer import train_CelebA_target_model
# 开启模型训练
train=True

# 数据集存储目录
root='E:/Project/ModelAndDataset/data'
# 模型训练存放目录
model_save_dir="E:/Project/ModelAndDataset/model/CelebA"
# 加载模型与攻击数据集
model_dir='E:/Project/ModelAndDataset/model/CelebA/CelebA_'
adv_dir='E:/Project/ModelAndDataset/data/celebA/Attack'
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_names=[
             'resnet18',
             'vgg19',
             'densenet169',
             'mobilenet'
             ]
attr_names = ['Eyeglasses','Wearing_Earrings','Male','Smiling','Wearing_Hat','Young']

batch_size=64
# 定义数据转换
transform=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
for model_name in model_names:
    # 定义模型
    if model_name=='resnet18':
        target_model=Target_model.ResNet18(2)
    elif model_name=='vgg19':
        target_model=Target_model.VGG_19(2)
    elif model_name=='densenet169':
        target_model=Target_model.Densenet169(2)
    elif model_name=='mobilenet':
        target_model=Target_model.MobileNet(2)
    # 加载模型
    for attr_name in attr_names:
        if train:
            if model_name=='resnet18' or model_name=='densenet169':
                optimizer='Adam'
            elif model_name=='vgg19' : 
                optimizer="SGD"
            elif model_name=='mobilenet':
                optimizer="RMSprop"

            train_CelebA_target_model(root=root,
                                save_dir=model_save_dir,
                                attr_name=attr_name,
                                model=model_name,
                                optimizer=optimizer
                                )
        target_model.load_state_dict(torch.load(model_dir+model_name+"_"+attr_name+".pth"))
        #加载对应属性的标签
        valid_dataset = CELEBA(root=root,
               train=False,
               train_ratio=0.98,
               transform=transform,
               label=attr_name,
               )
        clip_values = (0.0, 1.0)
        if model_name=='resnet18' or model_name=='densenet169':
            optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
        elif model_name=='vgg19':
            optimizer = torch.optim.SGD(target_model.parameters(), lr=0.0005, momentum=0.90)
        else:
            optimizer =torch.optim.RMSprop(target_model.parameters(),lr=0.001)
        estimator=PyTorchClassifier(model=target_model,loss=nn.CrossEntropyLoss(),
                                    optimizer=optimizer,
                                    input_shape=(3,64,64), nb_classes=2,clip_values=clip_values)
        Attack_Methods=[FastGradientMethod(estimator=estimator,eps=0.05),
                        BasicIterativeMethod(estimator=estimator,eps=0.05),
                        ProjectedGradientDescent(estimator=estimator,eps=0.05),
                        # PixelAttack(classifier=estimator,es=1),
                        # CarliniL2Method(classifier=estimator,confidence=0.5),
                        # ZooAttack(classifier=estimator,confidence=0.5)
                        ]
        for Attack_Method in Attack_Methods:
            attacker=Attack_Method
            generator=Adversarial_Examples_Generator(targetmodel=target_model,
                                         task=attr_name,
                                         dataset=valid_dataset,
                                         method=attacker,
                                         targeted=False,
                                         batch_size=64,
                                         save_dir=adv_dir,
                                         save_raw=True,
                                         device=device)
            #生成对抗样本并保存数据
            generator.generate()
            # 加载
            adv_example_dataset=CELEBA_Attack(root=adv_dir,
                                  adv=True,
                                  attackMethod=str(type(attacker).__name__),
                                  model=str(type(target_model)).split(".")[-1].split("'")[0],
                                  label=attr_name,
                                  transform=transform
                                )
            raw_example_dataset=CELEBA_Attack(root=adv_dir,
                                  adv=False,
                                  attackMethod=str(type(attacker).__name__),
                                  model=str(type(target_model)).split(".")[-1].split("'")[0],
                                  label=attr_name,
                                  transform=transform)
            
            adv_data_loader=torch.utils.data.DataLoader(adv_example_dataset,batch_size=64)
            raw_data_loader=torch.utils.data.DataLoader(raw_example_dataset,batch_size=64)
            try:
                advdata,advlabel=next(iter(adv_data_loader))
                rawdata,rawlabel=next(iter(raw_data_loader))
                drawCelebAImages(advdata,advlabel,
                            attr_name,
                            show=False,
                            save_path="E:/Project/ZLTProgram/CelebA/Attack_result/Adv"\
                            +adv_example_dataset.taskname+".png")
                drawCelebAImages(rawdata,rawlabel,
                            attr_name,
                            show=False,
                            save_path="E:/Project/ZLTProgram/CelebA/Attack_result/Raw"\
                            +raw_example_dataset.taskname+".png"
                            )
            except:
                print("AdvImage Number Error!")
                continue


    
       
         






