import torch
import torch.nn as nn
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent,BasicIterativeMethod
import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
from Model import Target_model
import torch.nn.functional as F
from Utils.MISC import *
import numpy as np
from Utils.AE_Util import Adversarial_Examples_Generator
from torch.utils.data import TensorDataset, DataLoader 
def drawConfusion_matrix(target_model_name,
                         attck_Method,
                         eps,   
                         selected_classes,                         
                         confusion_matrix,
                         save_path=None
                         ):
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Normal', 'Adversarial']  
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes,rotation=90)

    plt.xlabel('Predicted')
    plt.ylabel('Actual',rotation=90)

    # 在图表中添加数字标签
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(confusion_matrix[i, j]),fontsize=12,horizontalalignment="center", color="black")
    eps= "{:e}".format(eps).replace(".","")
    pic_name='{}_GTSRB_AT_training_{}_{}_{}'.format(target_model_name,len(selected_classes),attck_Method,eps)
    
    if save_path:
        plt.savefig(join(save_path,pic_name))


def train_GTSRB_at_model(root,
                        selected_classes,
                        num_epochs=10,
                        batch_size=128,
                        at_model_name='resnet18',
                        at_model_dir="F:\ModelAndDataset\model\GTSRB",
                        target_model_name='resnet18',
                        target_model_dir="F:\ModelAndDataset\model\GTSRB",
                        test_result_path=None,
                        criterion = F.cross_entropy,
                        Attack_method=FastGradientMethod,
                        train_eps=0.1,
                        test_eps=0.1,
                        device="cuda",
                        save=True):
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
    # 加载对抗训练模型
    at_save_path=at_model_dir+"/GTSRB_{}_{}_at.pth".format(at_model_name,len(selected_classes))
    # 加载模型
    if at_model_name=='resnet18':
        at_model=Target_model.ResNet18(num_classes=len(selected_classes)).to(device)
    elif at_model_name=='vgg19':
        at_model=Target_model.VGG_19(num_classes=len(selected_classes)).to(device)
    elif at_model_name=='densenet169':
        at_model=Target_model.Densenet169(num_classes=len(selected_classes)).to(device)
    elif at_model_name=='mobilenet':
        at_model=Target_model.MobileNet(num_classes=len(selected_classes)).to(device)
    # 定义优化器
    if at_model_name=='resnet18' or at_model_name=='densenet169':
        train_optimizer = torch.optim.Adam(at_model.parameters(), lr=0.001)
    elif at_model_name=='vgg19' : 
        train_optimizer = torch.optim.SGD(at_model.parameters(), lr=0.0005, momentum=0.90)
    elif at_model_name=='mobilenet':
        train_optimizer =torch.optim.RMSprop(at_model.parameters(),lr=0.001)

    clip_values = (0.0, 1.0)
    if os.path.exists(at_save_path) :
        print("File {} existed ,loading model!".format(at_save_path))
        at_model.load_state_dict(torch.load(at_save_path))
        at_model.to(device)
    else:
        #定义分类器
        at_model.to(device)
        at_classifier=PyTorchClassifier(model=at_model,loss=criterion,
                        optimizer=train_optimizer,
                        input_shape=(3,64,64), nb_classes=len(selected_classes),clip_values=clip_values)
        #定义使用对抗训练的攻击方法
        training_attacker=Attack_method(estimator=at_classifier,eps=train_eps)
        #===========进行对抗训练==============
        adv_trainer=AdversarialTrainer(classifier=at_classifier,attacks=training_attacker,ratio=0.5)
        at_model.train()
        for i,(images,labels) in enumerate(train_Loader):
            images=images.cpu().numpy()
            labels=mapping_labels(train_class_mapping,labels)
            labels=labels.unsqueeze(1).long()
            labels=labels.cpu().numpy()
            adv_trainer.fit(x=images,y=labels,batch_size=batch_size,nb_epochs=num_epochs)
    # 测试对抗训练模型的正常精度
    # 评估模型
    at_model.eval()
    correct = 0
    total = 0
    for images, labels in test_Loader:
        images=images.to(device)
        labels=mapping_labels(test_class_mapping,labels)
        outputs = at_model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on normal data: {accuracy:.2f}%,{correct}')
    
    #==========针对相应普通模型生成对抗样本===========
    # 加载测试用的普通目标模型
    target_save_path=target_model_dir+"/GTSRB_{}_{}.pth".format(at_model_name,len(selected_classes))
    # 加载模型
    if target_model_name=='resnet18':
        target_model=Target_model.ResNet18(num_classes=len(selected_classes)).to(device)
    elif target_model_name=='vgg19':
        target_model=Target_model.VGG_19(num_classes=len(selected_classes)).to(device)
    elif target_model_name=='densenet169':
        target_model=Target_model.Densenet169(num_classes=len(selected_classes)).to(device)
    elif target_model_name=='mobilenet':
        target_model=Target_model.MobileNet(num_classes=len(selected_classes)).to(device) 
    # 定义优化器
    if target_model_name=='resnet18' or target_model_name=='densenet169':
        test_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
    elif target_model_name=='vgg19' : 
        test_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.0005, momentum=0.90)
    elif target_model_name=='mobilenet':
        test_optimizer =torch.optim.RMSprop(target_model.parameters(),lr=0.001)  
   
    target_model.load_state_dict(torch.load(target_save_path))
    target_classifier=PyTorchClassifier(model=target_model,loss=nn.CrossEntropyLoss(),
                                optimizer=test_optimizer,
                                input_shape=(3,64,64), nb_classes=len(selected_classes),clip_values=clip_values)  
    testing_attacker=Attack_method(estimator=target_classifier,eps=test_eps)
    # 定义对抗样本生成器
    ae_generator=Adversarial_Examples_Generator(
            targetmodel=target_model,
            method=testing_attacker,
            targeted=False,
            batch_size=batch_size,
            save_dir=None,
            device=device
            )
    
    raw_imgs,adv_imgs,raw_labels,adv_labels=ae_generator.generate(test_Loader,test_class_mapping)
    rawDataset=TensorDataset(raw_imgs,raw_labels)
    advDataset=TensorDataset(adv_imgs,adv_labels)
    rawLoader=DataLoader(dataset=rawDataset,batch_size=batch_size,shuffle=False)
    advLoader=DataLoader(dataset=advDataset,batch_size=batch_size,shuffle=False)
    TP=0
    TN=0
    FP=0
    FN=0
    # 测试对抗训练的效果
    for (normal_imgs,normal_labels),(ae_imgs,ae_labels) in zip(rawLoader,advLoader):
        target_model.eval()
        target_model.to(device)
        at_model.eval()
        at_model.to(device)
        # 比较对抗训练模型的输出结果与正常样本的真实标签
        at_model_output=at_model(normal_imgs)
        _,at_predicted=at_model_output.max(1)
        # 计算TN，FP
        at_correct_normal=torch.eq(at_predicted,normal_labels)
        TN+=torch.sum(at_correct_normal).item()
        at_wrong_normal=~at_correct_normal
        FP+=torch.sum(at_wrong_normal).item()
        # 比较对抗训练模型的输出结果与对抗样本的真实标签
        at_model_output=at_model(ae_imgs)
        _,at_predicted=at_model_output.max(1)
        # 计算TP，FN
        at_correct_adv=torch.eq(at_predicted,normal_labels)
        TP+=torch.sum(at_correct_adv).item()
        at_wrong_adv=~at_correct_adv
        FN+=torch.sum(at_wrong_adv).item()
    confusion_matrix=np.array([[TN,FP],[FN,TP]])
    drawConfusion_matrix(at_model_name,
                         str(type(testing_attacker).__name__),
                         test_eps,
                         selected_classes,
                         confusion_matrix,
                         test_result_path
                         )

    if save:
        torch.save(at_model.state_dict(),at_save_path)
        print('Save Model to{}!'.format(at_save_path))
    



        









        






        
        



    
    
    
    

    
