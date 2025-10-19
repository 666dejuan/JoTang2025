import torch
import pandas as pd
from preprocessing import feature_engineering
from model import create_model

def load_and_preprocess_test_data(test_file_path='Titanic_data/test.csv'): # 加载并预处理测试数据
    # 加载测试数据
    test_data = pd.read_csv(test_file_path)

    # 保存PassengerId用于最终提交
    passenger_ids = test_data['PassengerId'].copy()

    # 使用与训练数据相同的预处理
    processed_test_data = feature_engineering(test_data)

    return processed_test_data, passenger_ids

def predict_test_data(model, processed_test_data): # 对测试数据进行预测

    model.eval()
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(processed_test_data.values)

    # 进行预测
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1) # 转化为概率分布
        predictions = torch.argmax(outputs, dim=1) # 取最大值的索引，选择概率最高的类别作为预测结果

    return predictions.numpy(), probabilities.numpy()


def create_submission_file(passenger_ids, predictions, output_file='submission.csv'): # 创建提交文件
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    submission_df.to_csv(output_file, index=False) # 将数据保存在submission.csv中
    print(f"提交文件已保存为: {output_file}")
    return submission_df


def compare_with_gender_submission(submission_df, gender_submission_file='Titanic_data/gender_submission.csv'): # 与gender_submission.csv进行比较
    try:
        gender_submission = pd.read_csv(gender_submission_file)
        merged_df = submission_df.merge(gender_submission, on='PassengerId', suffixes=('_pred', '_gender')) # 合并两文件，构建新的得df，on指定合并的列名，suffixes处理冲突，原来的Survived列变成两列并加上了后缀

        accuracy = (merged_df['Survived_pred'] == merged_df['Survived_gender']).mean() # mean是pandas的方法
        print(f"与gender_submission.csv比较的准确率: {accuracy:.4f}")

        return accuracy, merged_df
    except FileNotFoundError:
        print("未找到gender_submission.csv文件")
        return None, None


def main():
    # 加载训练好的模型
    print("正在加载训练好的模型...")
    model, _, _ = create_model()

    # 请先运行main.py训练模型，然后将模型保存下来
    try:
        model.load_state_dict(torch.load('titanic_model.pt')) # 加载训练好的模型
        print("模型加载成功!")
    except FileNotFoundError:
        print("未找到训练好的模型文件，请先运行main.py训练模型")
        return

    # 处理测试数据
    print("正在处理测试数据...")
    processed_test_data, passenger_ids = load_and_preprocess_test_data('Titanic_data/test.csv')

    # 进行预测
    print("正在进行预测...")
    predictions, probabilities = predict_test_data(model, processed_test_data)

    # 创建提交文件
    submission_df = create_submission_file(passenger_ids, predictions)

    # 与gender_submission.csv比较
    compare_with_gender_submission(submission_df)

    # 显示预测结果的统计信息
    print(f"\n预测结果统计:")
    print(f"生存预测: {predictions.sum()}/{len(predictions)} ({predictions.sum() / len(predictions) * 100:.2f}%)")

    # 打印真实生存率
    real_rate = pd.read_csv('Titanic_data/train.csv')['Survived'].mean() * 100
    print(f"真实生存率: {real_rate:.2f}%")

if __name__ == "__main__":
    main()