import csv
import json
import re


# 0.读取advbench的子集数据
def data_read_advbench_sub(data_path):
    # 创建一个字典来存储每个类别的bad_q集合
    categories = {}

    # 读取JSONL文件并分类bad_q
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            category = data["ss_category"]
            bad_q = data["bad_q"]

            if category not in categories:
                categories[category] = []
            categories[category].append(bad_q)

    # 提取所有类别中的bad_q，并放在一个新的集合中
    all_bad_qs = set()
    for bad_qs in categories.values():
        all_bad_qs.update(bad_qs)

    # 提取类别为Misrepresentation的元素放入一个新的集合中
    misrepresentation_bad_qs = set(categories.get("Misrepresentation", []))

    # 提取类别为Logical appeal的元素放入另一个新的集合中
    logical_appeal_bad_qs = set(categories.get("Logical appeal", []))

    print(f"all_data:{all_bad_qs}")
    print(f"misrepresentation_bad_qs:{misrepresentation_bad_qs}")
    print(f"logical_appeal_bad_qs:{logical_appeal_bad_qs}")

    return all_bad_qs, misrepresentation_bad_qs, logical_appeal_bad_qs


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

def data_pap_reader(data_path):
    with open(data_path, "r") as f:
        # 创建CSV阅读器
        csv_reader = csv.reader(f, delimiter='\n')
        # 读取CSV数据到列表
        list_data = [row for row in csv_reader if row]
        return list_data

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
    data_path = r"E:/code/Chemotherapy/data/adv_bench_sub_gpt4.jsonl"

    # 调用函数并获取结果
    data = data_read_advbench_sub(data_path)

    # # 打印所有bad_q
    # print("All bad_qs:")
    # for bad_q in data["all_bad_qs"]:
    #     print(bad_q)

    # # 打印Misrepresentation类别的bad_q
    # print("\nMisrepresentation bad_qs:")
    # for bad_q in data["misrepresentation_bad_qs"]:
    #     print(bad_q)

    # # 打印Logical appeal类别的bad_q
    # print("\nLogical appeal bad_qs:")
    # for bad_q in data["logical_appeal_bad_qs"]:
    #     print(bad_q)


if __name__ == "__main__":
    main()
