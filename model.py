import torch
from torch import nn
import numpy as np
import scipy
from torch.nn import functional as F

class GoogleNet(nn.Module):

    def __init__(self, **kwargs):
        super(GoogleNet, self).__init__(**kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Sequential(nn.Flatten())
        self.linear1 = nn.Sequential(
            nn.Linear(55696, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.output = nn.Sequential(nn.Linear(128, 10),
                                    nn.Softmax(dim = 0))
    
    def forward(self, x):
        shape = []
        out = self.conv1(x)
        shape.append(out.shape)
        out = self.conv2(out)
        shape.append(out.shape)
        out = self.flatten(out)
        shape.append(out.shape)
        out = self.linear1(out)
        shape.append(out.shape)
        out = self.linear2(out)
        shape.append(out.shape)
        out = self.output(out)
        shape.append(out.shape)

        return out



class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
'''
X = torch.rand(size=(1, 1, 96, 96))
model = GoogleNet()
out, shape = model(X)
print(shape)
'''