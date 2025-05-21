import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),

        )

    def forward(self, x):
        return self.model1(x)

if __name__ == '__main__': # 测试模型正确性
    mynet=Net1()
    input = torch.ones((64, 3, 32, 32))  # 创建给定大小的张量 值全部是1
    output = mynet(input)
    print(output.shape)


