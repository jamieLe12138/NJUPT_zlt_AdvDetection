import torch
from torchvision import datasets, transforms
from Target_model.Target_model import CNN_MNIST
import torch.nn as nn
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载和加载MNIST数据集
train_dataset = datasets.MNIST(root='E:/Project/ModelAndDataset/data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='E:/Project/ModelAndDataset/data', train=False, transform=transform)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model =CNN_MNIST().to(device)
#损失函数与优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs =15
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #print("Current batch:{}".format(i+1))
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(),"E:/Project/ModelAndDataset/model/mnist_cnn.pth")

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


