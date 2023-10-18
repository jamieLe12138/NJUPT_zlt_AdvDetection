
import torch.nn as nn
####### 定义生成器 Generator #####
class generator_mnist(nn.Module):
    def __init__(self,z_dimension):
        super(generator_mnist, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 128),  # 用线性变换将输入映射
            nn.ReLU(True),  # relu激活
            nn.Linear(128, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 512),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(512, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )
    def forward(self, x):
        x = self.gen(x)
        return x
class generator_mnistConv(nn.Module):
    def __init__(self, z_dimension):
        super(generator_mnistConv, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dimension + 10, 512, kernel_size=4, stride=1, padding=0),  # 转置卷积
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=3),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.gen(x)
        return x
