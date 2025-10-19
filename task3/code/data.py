import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(123)

class TopLeftCrop: #定义一个左上角裁剪类
    def __init__(self):
        pass
    def __call__(self, img):
        return img.crop((0, 0, 24, 24))

def create_dataloaders():
    # 1. 定义数据预处理
    train_tf = transforms.Compose([
        TopLeftCrop(),# 左上角24x24裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 从ImageNet数据集上计算得出的统计值，比较贴近自然图像的分布
    ])

    val_tf = transforms.Compose([
        TopLeftCrop(), # 左上角24x24裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集
    train_dataset = ImageFolder(
        root='custom_image_dataset/train',  # 训练集路径
        transform=train_tf  # 应用训练集的数据增强
    )

    val_dataset = ImageFolder(
        root='custom_image_dataset/val',  # 验证集路径
        transform=val_tf  # 应用验证集的预处理（无数据增强）
    )

    # 3. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,  # 训练时打乱数据
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,  # 验证时不打乱
        num_workers=0
    )

    return train_loader, val_loader