import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

#获取数据集
iris = datasets.load_iris()

# task1 查看数据相关属性
print("iris对象的所有属性:")
print(dir(iris))

print("数据集特征形状:", iris.data.shape)
print("目标值形状:", iris.target.shape)

# task2 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42)

print("训练集特征形状:", X_train.shape)
print("测试集特征形状:", X_test.shape)
print("训练集目标形状:", y_train.shape)
print("测试集目标形状:", y_test.shape)

# 将数据转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # 将目标值转换为PyTorch张量并调整形状,使用LongTensor用于多分类
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 定义线性回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_dim=iris.data.shape[1]
output_dim=len(np.unique(iris.target)) # 3个类别
model = LogisticRegressionModel(input_dim,output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 自动包含softmax
optimizer = optim.SGD(model.parameters(), lr=0.01) # model.parameters()获取模型参数，lr设置学习率

# 训练模型
epochs = 1000 # 训练周期
train_losses = [] # 创建空列表，用来存储损失值，也可用来绘制损失函数曲线

print("\n开始训练模型...")

for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad() # 将模型所有参数的梯度重置为0
    loss.backward() # 反向传播
    optimizer.step() # 参数更新

    train_losses.append(loss.item()) # 这里使用append方法添加元素

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("训练完成!")

# task3 在测试集上预测
model.eval() # 设置模型为评估模式
with torch.no_grad(): # 禁用梯度计算
    # 获取预测结果
    test_outputs = model(X_test_tensor) # 前馈
    # _通常表示不关心的变量（不再被使用），1表示dim=1，沿列方向操作，括号前后两值代表最大值和最大值索引，我们只需索引来预测类别
    _, predicted_tensor = torch.max(test_outputs, 1)
    # 由PyTorch张量转换成Numpy数组
    y_pred = predicted_tensor.numpy()

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型在测试集上的准确率: {accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)
