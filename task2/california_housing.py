import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
california_housing = fetch_california_housing()

# task1 查看数据相关属性
print("california_housing对象的所有属性:")
print(dir(california_housing))

print("数据集特征形状:", california_housing.data.shape)
print("目标值形状:", california_housing.target.shape)

# task2 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    california_housing.data,
    california_housing.target,
    test_size=0.3,
    random_state=42)

print("训练集特征形状:", X_train.shape)
print("测试集特征形状:", X_test.shape)
print("训练集目标形状:", y_train.shape)
print("测试集目标形状:", y_test.shape)

# 将数据转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1) # 将目标值转换为PyTorch张量并调整形状，1: 第二维度大小为1
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# 初始化模型
input_dim = X_train.shape[1] #由之前打印结果可知为8
model = LinearRegressionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # model.parameters()获取模型参数，lr设置学习率

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

    if (epoch + 1) % 100 == 0: # 使用epoch+1的原因是排除掉第0轮，0%100=0
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# task3 在测试集上预测和评估，计算MSE和R方
model.eval() # 设置模型为评估模式
with torch.no_grad(): # 禁用梯度计算
    y_pred_tensor = model(X_test_tensor) # 前向传播
    y_pred = y_pred_tensor.numpy().flatten() # 格式转化（相当重要）转化前是二维张量，转化后是一维数组，才能与真实值计算

# 计算模型的性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型性能评估:")
print("均方误差 (MSE):", mse)
print("决定系数 (R²):", r2)

# task4 绘图
plt.figure(figsize=(6, 5))

# 真实房价 vs 预测房价
# 画条基准红线，呈现完美的状况，y=x，预测值等于真实值
# 前两个[]确定了线的起止位置，r代表red，--代表虚线，lw代表线条粗细line width
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# 添加标签和标题
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('真实房价 vs 预测房价')

plt.show()