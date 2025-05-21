# conda list scipy 终端查看是否有scipy包

import torch
import torchvision.datasets
from hvplot import output
from torch import nn
from torch.nn import Conv2d  # !这是常规的卷积层
from torch.nn import MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from a_lossandbackup import loss_cross
from a_torchvision import target

# 加载模型代码 从class开始
VGG6_false=torchvision.models.vgg16(pretrained=False) # 随机初始化参数

# 加载模型代码和参数
VGG6_true=torchvision.models.vgg16(pretrained=True) # 使用imagenet预训练参数


# 如何利用现有网络模型(训练好其他数据集参数)应用到自己的数据集上
# out_features=1000 想改成10(CIFAR10)

'''
方法一: 最后加一个线性层
VGG6_true.add_module('add_linear',nn.Linear(1000,10))
'''

'''
方法二: 加在classifier(一个Sequential)里面
VGG6_true.classifier.add_module('add_linear',nn.Linear(1000,10))
'''

'''
# 方法三: 直接改classifier最后一层的Linear
VGG6_true.classifier[6]=nn.Linear(4096,10)
'''




