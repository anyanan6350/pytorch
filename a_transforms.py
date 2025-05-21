from dask.sizeof import sizeof
from torchvision import transforms #图片变换
from PIL  import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize

writer=SummaryWriter("logs") # 事件存在logs文件夹下

img_path="dataset/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)

# Totensor
tensor_tans=transforms.ToTensor() # ToTensor类的实例
img_tensor=tensor_tans(img_PIL) # PIL转换成tensor类型
writer.add_image("tensor_img",img_tensor,1)

# Normalize 图片格式归一化
# print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
# ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
img_norm=trans_norm(img_tensor)
# print(img_norm[0][0][0])
writer.add_image("normalize",img_norm)



# Resize
# print(img_PIL)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img_PIL)
img_resize=tensor_tans(img_resize) # 转换成tensor
writer.add_image("resize",img_resize,0)
# print(img_resize)



# Compose
trans_resize_2=transforms.Resize(512) # 只写小边等比缩放
trans_compose=transforms.Compose([trans_resize_2,tensor_tans]) #组合31 32行
# trans_resize_2的输出是tensor_tans的输入

img_resize_2=trans_compose(img_PIL) # 输入->输出
writer.add_image("resize",img_resize_2,1)
# print(img_resize_2.shape)


# RandomCrop 随机裁剪

trans_random=transforms.RandomCrop(512) #((512,1024))
trans_compose_2=transforms.Compose([trans_random,tensor_tans])
for i in range(10):
    img_crop=trans_compose_2(img_PIL)
    writer.add_image("random_crop",img_crop,i)


writer.close()



