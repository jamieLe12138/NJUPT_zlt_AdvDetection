import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch
import sys
sys.path.append("E:/Project/ZLTProgram/GTSRB")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
import Model.Target_model as Target_model
import torch
from Utils.MISC import *

def train_GTSRB_target_model(root,save_dir,
                              selected_classes,
                              pretrained_model_path=None,
                              save=True,
                              num_epochs=10,
                              batch_size=128,
                              model='resnet18',
                              criterion = F.cross_entropy,
                              optimizer = "Adam",
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

    save_path=save_dir+"/GTSRB_{}_{}.pth".format(model,len(selected_classes))
    if os.path.exists(save_path) and pretrained_model_path==None:
        print("File {} already existed ,skip trainning!".format(save_path))
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

        # 定义优化器
        if optimizer=="Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif optimizer=="SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.90)
        elif optimizer=="RMSprop":
            optimizer =torch.optim.RMSprop(model.parameters(),lr=0.001)
   
        for epoch in range(num_epochs):
            # 训练模型
            model.train() 
            for i,(images,labels) in enumerate(train_Loader):
                images=images.to(device)
                labels=mapping_labels(train_class_mapping,labels)
                labels=labels.unsqueeze(1).long()
        
                one_hot_labels=get_one_hot_label(labels,num_classes=len(selected_classes)).to(device)
                outputs = model(images).to(device)
        
                optimizer.zero_grad()
                loss = criterion(outputs, one_hot_labels)
                if(i%10==0) :
                    print(f'Batch:{i}/{len(train_Loader)},Loss:{loss.item():.4f}')
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
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
            print(f'Accuracy on test data: {accuracy:.2f}%,{correct}')
        
            #训练精度到达95%以上停止训练
            if accuracy>95 :
                print("Current Accuracy is up to 95%,stop trainning")
                break

            if accuracy>90 and accuracy<=95 :
                if epoch>10:
                    print("Current Accuracy is up to 90% ,current epoch is {},stop trainning".format(epoch))
                    break
                else:
                    print("Current Accuracy is up to 90% ,current epoch is {},continue trainning".format(epoch))

            if accuracy>86 and accuracy<=90 :
                if epoch>15:
                    print("Current Accuracy is up to 86% ,current epoch is {},stop trainning".format(epoch))
                    break
                else:
                    print("Current Accuracy is up to 86% ,current epoch is {},continue trainning".format(epoch))

            if accuracy<86:
                print("Current Accuracy too low,current epoch is {},continue trainning".format(epoch))

        if save:
            torch.save(model.state_dict(),save_path)
            print('Save Model to{}!'.format(save_path))




















