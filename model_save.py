import torch
import torchvision

VGG6_false=torchvision.models.vgg16(pretrained=False) # 随机初始化参数

# 保存方式1 模型+参数
torch.save(VGG6_false,"save_VGG16_method1.pth") # pth 保存格式

# 保存方式2 参数(推荐,小)
torch.save(VGG6_false.state_dict(),"save_VGG16_method2.pth") #保存参数字典形式