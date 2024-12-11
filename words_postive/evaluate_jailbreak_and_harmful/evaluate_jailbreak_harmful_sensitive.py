import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.special import (
    softmax,
)  # 如果使用 sklearn, 则为 from sklearn.preprocessing import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.detect_toxic_utils.sensitive_detect import (
    predict_sensitivity,
)  # 导入敏感性预测函数
from utils.data_util.data_utils import data_pap_reader  # 读取恶意输入


log_dir = "E:/code/Chemotherapy/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = os.path.join(log_dir, "role.log")

PAP_origin_Input_path = r"E:/code/Chemotherapy/data/PAP_origin_input.csv"
PAP_harmful_Input_path = r"E:/code/Chemotherapy/data/PAP_harmful_input.csv"


def main():
    # 1.读取两个不同的数据文件
    origin_data_list = data_pap_reader(PAP_origin_Input_path)
    harmful_data_list = data_pap_reader(PAP_harmful_Input_path)

    # 2.计算每个集合中所有元素的敏感分数
    harm_input_scores = []
    pap_input_scores = []
    for idx, origin_input in enumerate(
        tqdm(origin_data_list, desc="PAP Origin Test Data From HarmfulBench"), start=1
    ):
        origin_input_score = predict_sensitivity(origin_input)  # 原始敏感分数
        pap_input = harmful_data_list[idx - 1]
        pap_input_score = predict_sensitivity(pap_input)
        print(f"第{idx}条数据-{origin_input}的敏感分数为：{origin_input_score}")
        print(f"第{idx}条数据-{pap_input}的敏感分数为：{pap_input_score}")
        harm_input_scores.append(origin_input_score)
        pap_input_scores.append(pap_input_score)
        
    

    # 3.绘制图像
    # 使用对数坐标轴
    x_origin = range(1, len(harm_input_scores) + 1)
    x_pap = range(1, len(pap_input_scores) + 1)
    plt.figure(figsize=(10, 6))
    # plt.plot(origin_input_scores[:20], label='Origin Data', marker='o')
    # 绘制原始数据的敏感分数
    plt.plot(
        x_origin,
        harm_input_scores[:20],
        "r-",
        label="Harmful Input",
        marker="o",
        linewidth=2,
    )
    # 绘制经过pap数据的敏感分数
    plt.plot(
        x_pap,
        pap_input_scores[:20],
        "y-",
        label="PAP Input",
        marker="o",
        linewidth=2,
    )
    # plt.plot(pap_input_scores[:20], label='Harmful Data', marker='o')
    plt.title("Sensitivity scores for common sentence versus PAP sentence ")
    plt.xticks(
        range(1, len(harm_input_scores) + 1, 1),
        range(1, len(harm_input_scores) + 1, 1),
    )
    plt.xlabel("Data Index")
    plt.ylabel("Sensitivity Score")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
