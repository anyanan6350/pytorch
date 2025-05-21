import torchvision.datasets
import torch
from mpmath.identification import transforms
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10(root='./dataset1', train=False, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset, batch_size=64, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear = torch.nn.Linear(196608, 10)

    def forward(self, x):
        return self.Linear(x)

net = Net()

data_iter=iter(dataloader)
imgs,target=next(data_iter)
outputs=torch.flatten(imgs) #张量压扁成向量
outputs=net(outputs)
print(outputs.shape)



