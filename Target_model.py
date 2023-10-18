import torch
import torch.nn as nn
import torchvision.models as models
class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1=nn.Linear(28 * 28, 200)       
        self.fc2=nn.Linear(200, 200)
        self.fc3=nn.Linear(200, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x),dim=1)
        return x
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,5,1,2),
            nn.ReLU(),
            nn.Conv2d(64,64,5,2,0),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128,10),
            nn.Softmax()        
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class VGG_19(nn.Module):
    def __init__(self,num_classes):
        super(VGG_19,self).__init__()
        self.model=models.vgg19(pretrained=True)
        # 修改VGG19的最后一层以适应指定的类别数
        num_features=self.model.classifier[6].in_features
        self.model.classifier[6]=nn.Linear(num_features,num_classes)
    def forward(self,x):
        return self.model(x)
