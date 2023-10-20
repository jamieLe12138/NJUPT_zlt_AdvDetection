import art
from art.attacks.evasion import FastGradientMethod
from Target_model import MLP_MNIST
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#加载目标模型
# 使用你的 MLP 模型创建 CustomPyTorchEstimator
model=MLP_MNIST()
model.load_state_dict(torch.load(""))
model.to(device)
estimator=PyTorchClassifier(model=model,loss=nn.CrossEntropyLoss(),
                            optimizer = torch.optim.Adam(model.parameters(), lr=0.001),
                            input_shape=(1,28,28), 
                            nb_classes=10)



#定义攻击器
attacker = FastGradientMethod(estimator, eps=0.1)
#加载数据
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_test= datasets.MNIST(
    root='E:/Project/ModelAndDataset/data', train=False, transform=img_transform, download=True
)
batch_size = 64  # 选择适当的批次大小
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

#初始化
raw_img=torch.empty(0).to(device)
adv_img=torch.empty(0).to(device)
raw_labels = torch.empty(0).to(device)
adv_labels = torch.empty(0).to(device)
adv_probs = torch.empty(0).to(device)

for batch_idx, (data, label) in enumerate(test_loader):
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
    print("raw_predicted:",raw_predicted)
    print("adv_predicted:",adv_predicted)
    #判断是否成功攻击
    comparison = torch.eq(raw_predicted, adv_predicted).to(device)
    print("comparison:",comparison)
    #添加成功攻击的样本
    raw_img=torch.cat([raw_img,raw_data[~comparison]],dim=0)
    adv_img=torch.cat([adv_img,adv_data[~comparison]],dim=0)
    raw_labels = torch.cat([raw_labels, raw_predicted[~comparison].to(device)], dim=0)
    adv_labels = torch.cat([adv_labels, adv_predicted[~comparison].to(device)], dim=0)
    adv_probs = torch.cat([adv_probs, torch.softmax(adv_output[~comparison], dim=1).to(device)], dim=0)

print(raw_img.shape)
print(adv_img.shape)
print(raw_labels.shape)
print(adv_labels.shape) 
print(adv_probs.shape)   
    

