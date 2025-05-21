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

dataset=torchvision.datasets.CIFAR10(root='./dataset1', train=False, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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


mynet = Net()
# print(mynet)

'''
input=torch.ones((64,3,32,32)) # 创建给定大小的张量 值全部是1
output=mynet(input)
writer=SummaryWriter(log_dir='./logs1') #计算图
writer.add_graph(mynet,input)
writer.close()
'''


'''
data_iter=iter(dataloader) # 查看一个output和target的值
img,target=next(data_iter)
output=mynet(img)
print(output)
print(target)
'''

loss=nn.CrossEntropyLoss() # 多分类
optim=torch.optim.SGD(mynet.parameters(), lr=0.01) # 随机梯度下降
for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        output=mynet(imgs)
        result_loss=loss(output,targets) # output向量 targets标量
        optim.zero_grad()  # 每个batch梯度清零
        result_loss.backward() # 计算kennel梯度
        optim.step() # 优化参数
        running_loss=running_loss+result_loss.item() # 计算每轮梯度之和 正常的是下降
    print(running_loss)



