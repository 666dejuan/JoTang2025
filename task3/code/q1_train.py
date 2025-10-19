import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP冲突警告，最后遇到的问题

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import multiprocessing as mp

torch.manual_seed(123)

from data import create_dataloaders
from model import CNN

def main(): # 为了修复多进程问题，使所有代码都在 if __name__ == '__main__': 保护块内
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader = create_dataloaders()

    model = CNN()
    model.to(device)

    # 设置你的 training parameters
    num_epochs = 20 # 整个训练数据集将被模型完整地遍历多少遍
    lr = 0.001 # Adam优化器经典默认值
    weight_decay = 1e-4 # CNN常用，设置权重衰减

    # 设置你的 cross-entropy loss function
    loss_fn = nn.CrossEntropyLoss()

    # 设置你的优化器，注意用上你的 lr 和 weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 设置你的 learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # 阶梯式学习率调度器，每10个epoch调整一次学习率，学习率乘以0.1

    # 用于画图
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # 将图像和标签移动到设备上
            images = images.to(device)
            labels = labels.to(device)
            # 将梯度清零
            optimizer.zero_grad()
            # 通过模型进行前向传播
            outputs = model(images)
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 计算训练集每个 Epoch 的损失和准确率
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss_list.append(epoch_train_loss)
        train_accuracy_list.append(epoch_train_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        # 更新学习率：这里每个 epoch 调用一次 step，如果使用 per-batch 调度器，需要放到训练循环内部每个 batch 后
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # Compute validation loss and accuracy
            correct, total = 0, 0
            epoch_val_loss = 0.
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                # 向前传播
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                # 从模型输出中获取预测标签 label
                _,predicted =torch.max(outputs, 1)
                # 或predicted = torch.max(outputs, 1)[1]
                # 累加 correct 和 total
                # ==逐元素比较，返回布尔张量，.sum求和，.item()将单元素张量转换为python数值
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                epoch_val_loss += loss.item() * images.size(0) # 这里images.size(0)和labels.size(0)是相等的

            epoch_val_accuracy = correct / total
            epoch_val_loss /= total
            # 不直接计算loss_item的平均值是因为batch_size可能不一样，尤其是最后一个，样本数不满足设定的batch_size的整数倍
            val_loss_list.append(epoch_val_loss)
            val_accuracy_list.append(epoch_val_accuracy)

            epoch_list.append(epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

            # TODO: （可选）在这里，你可以保存效果最好的模型


    # 如果你之前没有保存模型，这里会保存最后一轮的模型状态
    torch.save(model.state_dict(), "q1_model.pt")

    # 绘制 training 和 validation 的 loss 和 accuracy 曲线
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(epoch_list, train_loss_list, label="Train")
    axs[0].plot(epoch_list, val_loss_list, label="Val")
    axs[0].set_yscale("log")

    axs[1].plot(epoch_list, train_accuracy_list, label="Train")
    axs[1].plot(epoch_list, val_accuracy_list, label="Val")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")

    for ax in axs:
        ax.legend()
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"q1_plots.png", dpi=300)
    plt.clf()
    plt.close()

# 添加主模块保护,遇到问题了
if __name__ == '__main__':
    # 对于Windows系统，可能需要
    mp.freeze_support()
    main()