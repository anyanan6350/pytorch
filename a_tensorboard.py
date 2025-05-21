import torch
from PIL  import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer=SummaryWriter("logs") # 事件存在logs文件夹下

img_path=r"dataset/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)

writer.add_image("test",img_array,1,dataformats='HWC') # 默认通道数在前,但是本数据在后面
# 换个tag 连个图框
# 更换地址 global_step=2 可以一个框展示两张图片
# 如果想展示所有图片?

# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data

for i in range(100):
    writer.add_scalar("y=2x",2*i,i) #添加表

writer.close()

#每运行一次生成一个新事件 在网页中刷新获得新图像

'''
tensorboard 报错原因
1. 没运行代码=没创建事件 
2. tensorboard --logdir=logs --port=6007 不要写两个等号
3. writer.close() 必须写在最后保存
'''
