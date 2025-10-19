import torch
from sklearn.metrics import accuracy_score,f1_score

def train_epoch(model, train_loader, optimizer,criterion):
    model.train() # 将模型设置为训练模式
    total_loss = 0 # 初始化累计损失值
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x) # 清空上一轮的梯度，防止梯度累积
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader) # 返回该epoch的平均损失

def evaluate_model(model,val_loader,):
    model.eval() # 将模型设置为评估模式
    all_preds, all_labels = [], []
    with torch.no_grad(): # 禁用梯度计算，节省内存和计算资源
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1) # 获取预测类别（取最大概率的索引）
            all_preds.extend(preds.numpy()) # 将预测结果转换为numpy并添加到列表
            all_labels.extend(batch_y.numpy()) # 将真实标签转换为numpy并添加到列表

    f1 = f1_score(all_labels, all_preds, average='weighted')  # 使用加权平均
    return accuracy_score(all_labels, all_preds),f1 # 计算并返回准确率

def train_model(model, train_loader, val_loader, optimizer, criterion,epochs):
    train_losses,val_accuracies,val_f1_scores = [],[],[] # 初始化训练损失和验证准确率的记录列表
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        val_accuracy,val_f1_score = evaluate_model(model,val_loader)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1_score)

        if((epoch+1)%50==0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Test Accuracy: {val_accuracy:.4f}, Val F1-Score: {val_f1_score:.4f}')

    return train_losses, val_accuracies, val_f1_scores