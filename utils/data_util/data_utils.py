import csv
import json
import re


# 1.读取有害行为数据集
def data_reader(data_path):
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过CSV文件的第一行（通常是列名）
        goals = []
        for row in reader:
            goal = row[0] + "."  # 获取每一行的第一个元素
            goals.append(goal)  # 将目标添加到列表中
    return goals  # 返回所有目标的列表


# 2.读取Chemotherapy生成的越狱数据，或由LLMs生成的响应
def jailbroken_data_reader(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)  # 加载JSON文件内容
    return data  # 返回解析后的JSON数据


# 3.预处理GPT-3.5-turbo生成的缩短句子候选
def remove_number_prefix(sentence):
    """
    移除句子开头的数字前缀。

    参数:
    sentence (str): 输入的句子字符串。

    返回:
    str: 移除数字前缀后的句子字符串。

    示例:
    >>> remove_number_prefix("1. This is a test.")
    'This is a test.'
    >>> remove_number_prefix("2. Another example.")
    'Another example.'
    """
    return re.sub(
        r"^\d+\.\s*", "", sentence
    )  # 使用正则表达式移除句子开头的数字和点号及后续空格


def main():
    # CSV文件路径
    csv_file_path = r"E:/code/Chemotherapy/data/harmful_behaviors.csv"

    # 读取CSV文件
    goals = data_reader(csv_file_path)
    print("Goals from CSV file:")
    for goal in goals:
        print(goal)


if __name__ == "__main__":
    main()
