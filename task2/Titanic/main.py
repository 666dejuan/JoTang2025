import torch
import numpy as np
import random
import pandas as pd
from preprocessing import feature_engineering, get_data_loader
from model import create_model
from train_and_eval import train_model
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed) # Python内置随机数
    np.random.seed(seed) # Numpy随机数
    torch.manual_seed(seed) # PyTorch CPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # PyTorch GPU随机数
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    torch.backends.cudnn.deterministic = True # 确保CUDA卷积操作确定性
    torch.backends.cudnn.benchmark = False # 关闭CUDA基准优化

def main():
    set_seed(123)
    # 加载数据
    train_data = pd.read_csv('Titanic_data/train.csv')

    # 数据预处理
    print("正在进行数据预处理...")
    processed_train_data = feature_engineering(train_data)

    # 获取数据加载器
    train_loader, val_loader = get_data_loader(processed_train_data)

    # 创建模型
    print("正在创建模型...")
    model, criterion, optimizer = create_model()

    # 训练模型
    print("开始训练模型...")
    train_losses, val_accuracies,val_f1_scores = train_model(
        model, train_loader, val_loader, optimizer, criterion,epochs=500
    )

    # 保存训练好的模型
    torch.save(model.state_dict(), 'titanic_model.pt')
    print("模型已保存为: titanic_model.pt")

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses) # x默认为索引
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(val_f1_scores)
    plt.title('Val F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')

    plt.tight_layout() # 自动调整布局
    plt.show()

    print(f"最终测试准确率: {val_accuracies[-1]:.4f}")


if __name__ == "__main__":
    main()