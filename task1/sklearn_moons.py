import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(123)

# step1 数据的准备
X,y = make_moons(n_samples=1000, shuffle=True, noise=0.1,random_state=123) # 获取原始数据，得到numpy数组

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) # 划分数据集

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test) # 将特征数据由numpy数组转换成torch张量

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test) # 将标签数据由numpy数组转换成torch张量

# step2 构建简单神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# step3 实例化模型，构架优化器和损失函数
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
accuracies = [] # 创建空列表，为画图做准备

# step4 训练模型
for epoch in range(500):
    y_pred = model(X_train) # 前馈，计算预测值
    loss = criterion(y_pred, y_train) # 计算损失函数值

    optimizer.zero_grad() # 过往梯度清零，防止累加
    loss.backward() # 反向传播
    optimizer.step() # 权重更新

    with torch.no_grad(): # 禁用梯度计算，节省内存
        test_outputs = model(X_test) #
        _,predicted = torch.max(test_outputs, dim=1)
        accuracy = (predicted == y_test).float().mean()

        losses.append(loss.item())
        accuracies.append(accuracy.item()) # 这里使用append方法添加元素，也是为画图做准备

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}') # 使用epoch+1的原因是排除掉第0轮，0%50=0

# step5 绘制损失函数和准确率函数图像
plt.figure(figsize=(12, 4)) # 设置窗口大小，单位英寸

plt.subplot(1, 2, 1) # 创建子图网格，参数分别表示有一行图，两列图（就两个图），激活第一个图（左到右），绘制第一幅
plt.plot(losses) # X轴自动使用索引（0,1,2,...），Y轴使用损失值losses
plt.title('Training Loss') # 打印标题
plt.xlabel('Epoch') # 打印x轴标签
plt.ylabel('Loss') # 后面的同理

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout() # 自动调整子图之间的间距，防止标签重叠，使布局更紧凑美观
plt.show() # 展示图像

# step6 绘制热力图
# 创建网格点来覆盖整个数据空间
def plot_decision_boundary(model, X, y, title): # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5 # 确定网格范围

    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                            torch.linspace(y_min, y_max, 100)) # 创建网格

    grid_points = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1) # 将网格点展平
    with torch.no_grad():
        probs = model(grid_points)
        _, predictions = torch.max(probs, 1) # 预测

    zz = predictions.reshape(xx.shape) # 重塑预测结果用于绘图

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.RdYlBu) # 绘制填充等高线图，设置透明度为30%，使用红黄蓝配色方案
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu) # 给点添加黑色边框，提高可视性
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

print("训练集决策边界:")
plot_decision_boundary(model, X_train, y_train, "Training Set Decision Boundary")
print("测试集决策边界:")
plot_decision_boundary(model, X_test, y_test, "Test Set Decision Boundary")