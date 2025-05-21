import torch
import torchvision

'''
# 对应保存方式1的加载
model=torch.load("save_VGG16_method1.pth")
自己定义的模型在另一个文件中 不用实例化
'''


# 对应保存方式2的加载
VGG6_false=torchvision.models.vgg16(pretrained=False) # 随机初始化参数
VGG6_false.load_state_dict(torch.load("save_VGG16_method2.pth")) #加载参数
# 选择自己想要训练的模型参数
# save_VGG16_method2.pth是路径!
