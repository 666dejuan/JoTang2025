import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

# 创建特征工程函数，对数据进行预处理
def feature_engineering(data):
    df = data.copy() # 复制数据

    # 填补缺失
    df['Age'] = df['Age'].fillna(df['Age'].median()) # 对没有年龄数据的样本进行填充，填充为所有年龄的中位数
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # 对没有港口数据的样本进行填充，填充为出现最多的港口名，若众数相等，填第一个

    # 新建特征列
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # 将家庭成员的相关数据合并，+1是加本人
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int) # 创建是否独身数据，将布尔值转化为整数
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.',expand=False) # 对Name列的每个字符串进行操作,提取匹配表达式的部分,空格加多个大小写字母和逗号
    # expand用于返回Series，可直接赋值给新列，若为True，则返回DataFrame，需要df['Title'] = result[0]语句
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')  # 稀有称呼归为一类
    df['Title'] = df['Title'].replace('Mlle', 'Miss')  # 统一称呼，把Mlle替换为Miss
    df['Title'] = df['Title'].replace('Ms', 'Miss') # 同理
    df['Title'] = df['Title'].replace('Mme', 'Mrs') # 同理

    # 删除无用特征列
    df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
    # axis=1，指定删除为列，若inplace=False（默认），则计算出新的df,原df不变，需进行赋值操作

    # 编码分类变量
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked']) # fit学习所有可能的类别，transform，将文字转化为数字
    df['Title'] = le.fit_transform(df['Title'])
    return df

# 创建获取数据加载器的函数
def get_data_loader(processed_data):

    # 特征和目标分离
    X = processed_data.drop(['Survived'], axis=1)
    y = processed_data.Survived

    # 标准化特征，使每个特征的数据分布转换为均值为0，标准差为1的标准正态分布，消除不同特征量纲影响
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆封数据集 先拆分再转化成Pytorch张量，提高内存效率
    X_train,X_test,y_train,y_test = train_test_split( # 注意数据顺序，先两X，再两y
        X_scaled,
        y.values,
        test_size=0.2,
        random_state=123,
        stratify=y) # 保证数据拆分后的分布保持一致，及使存活率大致相等

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test) # 获取y的原始数据值而不包含索引

    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader
