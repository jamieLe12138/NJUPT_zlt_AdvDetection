import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from Target_model import VGG_19
import torch
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 下载和加载CIFAR10数据集
train_dataset = datasets.CIFAR10(root='E:/Project/ModelAndDataset/data',train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='E:/Project/ModelAndDataset/data', train=False, transform=transform,download=True)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model=VGG_19(num_classes=10).to(device)
#损失函数与优化器
criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.90)
num_epochs =15
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        if(i%200==0) :
            print("Current batch:{}".format((i)))
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(),"E:/Project/ModelAndDataset/model/cifar10_vgg19.pth")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images=images.to(device)
        outputs = model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')
