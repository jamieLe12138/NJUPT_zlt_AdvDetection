import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch
import sys
sys.path.append("E:/Project/ZLTProgram/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
from Target_model import Target_model

#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from dataload import CELEBA,get_one_hot_label


# 训练属性
attr_name='Smiling'
# 定义数据转换
transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor()])


# 加载数据集
Dataset = CELEBA(root='E:/Project/ModelAndDataset/data', train=True,label=attr_name, transform=transform)
train_ratio = 0.7
dataset_size = len(Dataset)
train_size = int(train_ratio * dataset_size)
valid_size = dataset_size - train_size
train_dataset = torch.utils.data.Subset(Dataset, range(train_size))
valid_dataset = torch.utils.data.Subset(Dataset, range(train_size, train_size + valid_size))

batch_size=128

train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_Loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)



# model=Target_model.VGG_19(num_classes=1).to(device)
# save_path="E:/Project/ModelAndDataset/model/CelebA_vgg19_{}.pth".format(attr_name)

save_path="E:/Project/ModelAndDataset/model/CelebA_resnet18_{}.pth".format(attr_name)
model=Target_model.ResNet18(num_classes=2).to(device)

#损失函数与优化器
criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs =3

for epoch in range(num_epochs):
    model.train()
    for i,(images,labels) in enumerate(train_Loader):
        images=images.to(device)
        labels=labels.unsqueeze(1).long()
        
        one_hot_labels=get_one_hot_label(labels).to(device)
        outputs = model(images).to(device)
        
        optimizer.zero_grad()
        loss = criterion(outputs, one_hot_labels)
        if(i%200==0) :
            print(f'Batch:{i}/{train_size//batch_size+1},Loss:{loss.item():.4f}')
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_Loader:
        images=images.to(device)
        outputs = model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%,{correct}')



torch.save(model.state_dict(),save_path)






