import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input= torch.tensor([1,2,3],dtype=torch.float32).reshape(1,1,1,3)

target= torch.tensor([1,2,5],dtype=torch.float32).reshape(1,1,1,3)
# [[[[1., 2., 5.]]] 4维 batch_size=1 channel=1 row=1 line=3

loss=L1Loss(reduction='sum')
# reduction='mean' 默认  result = tensor(0.6667) L1=(1-1)+(2-2)+(5-3)/3=2/3
# reduction='sum' result = tensor(2.)

loss_mse=MSELoss(reduction='sum')
# reduction='mean' 默认  result = tensor(1.3333) MSE=(1-1)^2+(2-2)^2+(5-3)^2/3=4/3
# reduction='sum' result = tensor(4.)

x=torch.tensor([0.1,0.2,0.3]).reshape(1,3) # row=1 line=3
y=torch.tensor([1])

loss_cross=CrossEntropyLoss()
# 输出=各自类别概率 target=类别标签

result=loss_cross(x,y)
# loss越小=x是1的概率越大

print(result) # tensor(1.1019)

# !注意损失函数输入和输出的格式
