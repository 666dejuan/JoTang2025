import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        # 不用Softmax是因为CrossEntropyLoss内置了

    def forward(self, x):
        x = self.dropout(self.bn1(self.relu(self.fc1(x))))
        x = self.dropout(self.bn2(self.relu(self.fc2(x))))
        x = self.dropout(self.bn3(self.relu(self.fc3(x))))
        x = self.output(x)

        return x

# 创建模型，损失函数和优化器
def create_model(learning_rate=0.01, weight_decay=0.001):
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay# L2正则化相关参数
    )
    return model, criterion, optimizer