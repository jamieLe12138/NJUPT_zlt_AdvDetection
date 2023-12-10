import torch
import torch.nn as nn
import torchvision.models as models

class VGG_19(nn.Module):
    def __init__(self,num_classes):
        super(VGG_19,self).__init__()
        self.model=models.vgg19(weights=None)
        # 修改VGG19的最后一层以适应指定的类别数
        num_features=self.model.classifier[6].in_features
        self.model.classifier[6]=nn.Linear(num_features,num_classes)
    def forward(self,x):
        return self.model(x)

class Densenet169(nn.Module):
    def __init__(self,num_classes):
        super(Densenet169,self).__init__()
        self.model=models.densenet169(weights=None,num_classes=num_classes)
    def forward(self,x):
        return self.model(x)
    
class ResNet18(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18,self).__init__()
        self.model=models.resnet18(weights=None)
        # 获取 ResNet-18 模型的最后一层
        num_features = self.model.fc.in_features
        # 替换最后一层全连接层
        self.model.fc = nn.Linear(num_features, num_classes)
    def forward(self,x):
        return self.model(x)

class MobileNet(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet,self).__init__()
        self.model=models.mobilenet_v2(weights=None)
        num_features = self.model.classifier[1].in_features
        classifier = nn.Sequential(
            nn.Linear(num_features, 512),  # 添加一个或多个隐藏层，根据需要
            nn.ReLU(),
            nn.Linear(512, num_classes)  # 输出层的输出维度与新的类别数量匹配
                )
        # 替换MobileNetV2的分类头
        self.model.classifier = classifier
    def forward(self,x):
        return self.model(x)
    

