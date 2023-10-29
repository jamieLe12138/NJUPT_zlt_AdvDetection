import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch
import sys
sys.path.append("E:/Project/ZLTProgram/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
from Target_model import Target_model
import torch
from dataload import CELEBA,get_one_hot_label

def train_CelebA_target_model(root,save_dir,attr_name,
                              pretrained_model_path=None,
                              save=True,
                              train_ratio=0.7,
                              num_epochs=10,
                              batch_size=128,
                              model='resnet18',
                              shuffle=False,
                              criterion = F.cross_entropy,
                              optimizer = "Adam",
                              device="cuda"):
    # 定义数据转换
    transform=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
    # 加载数据集
    # Dataset = CELEBA(root=root, train=True,label=attr_name, transform=transform)
    # dataset_size = len(Dataset)
    # train_size = int(train_ratio * dataset_size)
    # valid_size = dataset_size - train_size
    train_dataset = CELEBA(root=root, train=True ,train_ratio=train_ratio,label=attr_name,transform=transform)
    valid_dataset = CELEBA(root=root, train=False,train_ratio=train_ratio,label=attr_name,transform=transform)
    train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_Loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    save_path=save_dir+"/CelebA_{}_{}.pth".format(model,attr_name)
    # 加载模型
    if model=='resnet18':
        model=Target_model.ResNet18(num_classes=2).to(device)
    elif model=='vgg19':
        model=Target_model.VGG_19(num_classes=2).to(device)
    elif model=='densenet169':
        model=Target_model.Densenet169(num_classes=2).to(device)
    elif model=='mobilenet':
        model=Target_model.MobileNet(num_classes=2).to(device)

    if pretrained_model_path:
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
            labels=labels.unsqueeze(1).long()
        
            one_hot_labels=get_one_hot_label(labels).to(device)
            outputs = model(images).to(device)
        
            optimizer.zero_grad()
            loss = criterion(outputs, one_hot_labels)
            if(i%200==0) :
                print(f'Batch:{i}/{train_dataset.size//batch_size+1},Loss:{loss.item():.4f}')
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 评估模型
        model.eval()
        correct = 0
        total = 0
        maxAcc_count=0
        max_acc=0
        for images, labels in test_Loader:
            images=images.to(device)
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy on test data: {accuracy:.2f}%,{correct}')
        
        if accuracy>max_acc:
            max_acc=accuracy
            maxAcc_count=0
        else:
            maxAcc_count+=1
        #训练精度到达90%以上或精度开始下降，停止训练
        if accuracy>90 and maxAcc_count==1 :
            break
        elif accuracy>87 and accuracy<=90 and maxAcc_count==2 :
            break 
        elif maxAcc_count>=3:
            break

    if save:
        torch.save(model.state_dict(),save_path)
        print('Save Model to{}!'.format(save_path))




















