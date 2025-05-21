import torch
import torchvision.datasets # torchvision下就有封装好的module
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

dataset=torchvision.datasets.CIFAR10(root='./dataset1', train=False, transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64,shuffle=True)

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3,stride=1,padding=0)
        # in_channels 输入通道数
        # out_channels 输出通道数
        # kernel_size = (5,5) 卷积核尺寸
        # stride=1
        # padding=0
        # padding_mode('zeros') 默认用0填充

        self.pool = nn.MaxPool2d(3,2,ceil_mode=True) # maxpool 只能操作浮点数!
        # ceil_mode = True 保留边缘不足池化核size的

        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        output = self.conv1(input)
        output = self.pool(output)
        # output = self.relu(output)
        output = self.sigmoid(output)
        return output

mynet1 = mynet()
# print(mynet1) # 查看网络结构

writer=SummaryWriter("logs1")
step=0
for data in dataloader:
    imgs,targets=data
    # img_grid = make_grid(imgs)
    output=mynet1(imgs)
    # out_grid=make_grid(output)

    # writer.add_images('input3',imgs,step)
    writer.add_images('sigmoid',output,step)
    step+=1

# add_images:  imput_format=[64,3,32,32]
# add_image :  imput_format=[3,32,32]
# img_grid = make_grid(imgs) fit add_image 拼接成网格 和 add_images 有区别


writer.close()

