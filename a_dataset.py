import torch
import os
from PIL import Image # 读取图片
from torch.utils.data import Dataset # 获取数据及其标签;数据量
from torch.utils.tensorboard import SummaryWriter



# ctrl+/ 单行注释 
# # 按住ctrl 点击函数help


'''  
# Image show
from PIL import Image # 读取图片 
img_path= r"E:\testtest\dataset\train\ants_image\0013035.jpg" # r "绝对地址" \被当成转移符号
img=Image.open(img_path)
img.show()
'''

'''
# Image name
import os
dir_path= "dataset/train/ants_image" # 相对地址就不会出现绝对地址的问题
img_path_list=os.listdir(dir_path)  # 将dir_path文件夹下的文件的名字放在一个列表里
img_path_list[0] # 0013035.jpg
'''

'''
# Image path 
root_dir=r"dataset\train"
label_dir="ants_image"  # ants_image 文件夹名称就是label
img_name=img_path_list[0]
path=os.path.join(root_dir,label_dir,img_name) #拼接地址
path # dataset\train\ants_image
'''    


class mydata(Dataset):# extend Dataset
    def __init__(self,root_dir,label_dir)  : # 必须重写
        self.root_dir=root_dir # dataset/train
        self.label_dir=label_dir # ants_image 文件夹名称就是label
        self.path=os.path.join(self.root_dir,self.label_dir) #拼接地址
        self.img_path_list=os.listdir(self.path)

    def __getitem__(self, index) :# 可选择重写
        img_name=self.img_path_list[index]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label 
    
    def __len__(self) :
        return len(self.img_path_list) # 返回列表长度
      
root_dir=r"dataset\train"
ants_label_dir="ants_image"
bees_label_dir="bees_image"
ants_dataset=mydata(root_dir,ants_label_dir) # 创建实例
bees_dataset=mydata(root_dir,bees_label_dir) # 创建实例

img,label=ants_dataset[0]
img.show()
len(ants_dataset)

train_dataset=ants_dataset+bees_dataset # gather


