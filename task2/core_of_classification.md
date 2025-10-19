# 分类
---
## 一、分类任务的输出contrast to regression

| 任务类型 | 输出本质 | 回答的问题 |
|---------|----------|------------|
| **分类** | 离散类别标签 | "是什么" |
| **回归** | 连续数值 | "是多少" |

本质不同：输出的是类别还是数值
## 2. 二分类 vs 多分类

### 二分类
- **定义**：输出只有两种可能的类别
- **例子**：垃圾邮件检测（垃圾/非垃圾）
- **实现**：1个输出节点 + **Sigmoid**

### 多分类  
- **定义**：输出有三种或以上可能的类别
- **例子**：手写数字识别（0-9）
- **实现**：K（类别数）个输出节点 + **Softmax**

## 3. 分类模型评价指标

### 核心指标

* **TP (True Positive)**：真正例（实际为正，预测为正）
* **TN (True Negative)**：真负例（实际为负，预测为正）
* **FP (False Positive)**：假正例（实际为正，预测为负）
* **FN (False Negative)**：假负例（实际为负，预测为负）

- **准确率**Accuracy：所有预测中正确的比例
**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **精确率**Precision：所有被预测为正类的样本中，确实是正类的比例
**Precision** = TP / (TP + FP)
- **召回率**Recall：所有真实的正类样本中，被正确预测出来的比例
**Recall** = TP / (TP + FN)
- **混淆矩阵**：分类结果详情表
==以鸢尾花分类为例==
$$
\begin{bmatrix}
19 & 0 & 0 \\
0 & 12 & 1 \\
0 & 0 & 13
\end{bmatrix}
$$
* 行：真实类别（实际是什么）
* 列：预测类别（模型认为是什么）
```py
# Iris数据集的默认顺序
['setosa', 'versicolor', 'virginica']
```
 第一行：真实setosa类别（19个样本）
- **19**：19个setosa被正确预测为setosa（True Positive for setosa）
- **0**：0个setosa被错误预测为versicolor
- **0**：0个setosa被错误预测为virginica

 第二行：真实versicolor类别（13个样本）
- **0**：0个versicolor被错误预测为setosa
- **12**：12个versicolor被正确预测为versicolor（True Positive for versicolor）
- **1**：1个versicolor被错误预测为virginica（False Positive for virginica）

 第三行：真实virginica类别（13个样本）
 - **0**：0个virginica被错误预测为setosa
- **0**：0个virginica被错误预测为versicolor
- **13**：13个virginica被正确预测为virginica（True Positive for virginica）
