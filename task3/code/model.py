import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=7,padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 80, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(80, 10)
    
    def forward(self, x, intermediate_outputs=False):
        # 存储中间层输出
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = None, None, None, None, None

        # 第一层卷积块
        x = self.conv1(x)
        conv1_out = x  # 保存卷积层输出
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        conv2_out = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        conv3_out = x
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        conv4_out = x
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        conv5_out = x

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 全连接层
        final_out = self.fc(x)
        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
