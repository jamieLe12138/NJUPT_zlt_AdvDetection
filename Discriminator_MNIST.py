import torch.nn as nn
# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
class discriminator_mnist(nn.Module):
    def __init__(self):
        super(discriminator_mnist, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.dis(x)
        return x
class discriminator_mnistConv(nn.Module):
    def __init__(self):
        super(discriminator_mnistConv, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 输入为单通道图像
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # 将特征图展平
            nn.Linear(128*7*7, 10),  # 最后一层全连接层
            nn.Softmax()
        )

    def forward(self, x):
        x = self.dis(x)
        return x