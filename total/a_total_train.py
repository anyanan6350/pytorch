import torchvision
from torch.utils.data import DataLoader # !加载器来源非常重要
from torch.utils.tensorboard import SummaryWriter

from total.a_total_model import *
import time
#准备数据集
train_data=torchvision.datasets.CIFAR10(root='./dataset1', train=True, transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10(root='./dataset1', train=False , transform=torchvision.transforms.ToTensor())

train_data_size=len(train_data)
test_data_size=len(test_data)
print("the length of train_data_size:{}".format(train_data_size))
print("the length of test_data_size:{}".format(test_data_size))

train_dataloader=DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_dataloader=DataLoader(dataset=test_data, batch_size=64, shuffle=True)


#网络
mynet=Net1()

#损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(mynet.parameters(), lr=learning_rate)

#训练
total_train_step=0
total_test_step=0
epoch=10

# #添加tensorboard
writer=SummaryWriter(log_dir='./total/total_log') # 当前位于E:/testtest
start=time.time() #开始计时
for i in range(epoch):
    print("--------NO.{} begin--------".format(i+1))

    #训练
    mynet.train() # 有 dropout batchnorm 层 才调用
    for data in train_dataloader:
        images, labels = data
        outputs = mynet(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        if total_train_step%100==0: # 每个batch都输出太多
            end=time.time()
            print(end-start) #统计运行一百次时间
            print("train NO.{}, loss {}".format(total_train_step,loss.item()))
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    #测试
    mynet.eval() # 有 dropout batchnorm 层 才调用
    total_test_loss=0
    total_acc=0 # 衡量分类质量
    with torch.no_grad(): # 不需要优化
        for data in test_dataloader:
            images, labels = data
            outputs = mynet(images)
            loss = loss_fn(outputs, labels)
            total_test_loss+=loss.item()
            acc=(outputs.argmax(1) == labels).sum()# 1: 横轴 一行一个x
            # argmax(1) 找到每个x向量中最大值的下标=输出 再和真实标签做对比 sum=所有true(1)的和
            total_acc+=acc
    print("total test loss:{}".format(total_test_loss))
    print("total test accuracy:{}".format(total_acc/test_data_size))
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_accuracy',total_acc/test_data_size,total_test_step)
    total_test_step+=1





    torch.save(mynet.state_dict(),'total_model_{}.pth'.format(i))

writer.close()

