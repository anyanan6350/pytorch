import torchvision # 包含很多数据集
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])


train_set=torchvision.datasets.CIFAR10(root='./dataset1', train=True, transform=dataset_transform, download=True) # 训练集

test_set=torchvision.datasets.CIFAR10(root='./dataset1', train=False, transform=dataset_transform, download=True) # 测试集



# img,target=test_set[0] # 第一个图片及其类别
# print(img)
# print(test_set.classes[target]) # cat

# 如果想展示所有图片?
writer=SummaryWriter("logs1")
for i in range(10):
    img,target=train_set[i]
    writer.add_image('test_set',img,i)

writer.close()