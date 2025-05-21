import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from a_torchvision import target

test_data=torchvision.datasets.CIFAR10("./dataset1", train=False, transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data, batch_size=4, shuffle=True,num_workers=0,drop_last=False)
#DataLoader通过设置参数选取batch
#drop_last=False最后不足4的不舍弃

'''
img,target=test_data[0]
print(img.shape)
print(target)
'''

writer=SummaryWriter("logs1")
for epoch in range(2): # 两轮全体
    step=0
    for data in test_loader:
        imgs,targets=data
        # print(imgs.shape) # torch.Size([4, 3, 32, 32]) 4张图片
        # print(targets) # tensor([5, 0, 7, 7]) 随机采样
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step+=1
#全选 tab 缩进
writer.close()