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
    origin_input_scores = []
    pap_input_scores = []
    for idx, origin_input in enumerate(
        tqdm(origin_data_list, desc="PAP Origin Test Data From HarmfulBench"), start=1
    ):
        origin_input_score = predict_sensitivity(origin_input)  # 原始敏感分数
        pap_input = harmful_data_list[idx - 1]
        pap_input_score = predict_sensitivity(pap_input)
        print(f"第{idx}条数据-{origin_input}的敏感分数为：{origin_input_score}")
        print(f"第{idx}条数据-{pap_input}的敏感分数为：{pap_input_score}")
        origin_input_scores.append(origin_input_score)
        pap_input_scores.append(pap_input_score)
        
    

    # 3.绘制图像
    x_origin = range(1, len(origin_input_scores) + 1)
    x_pap = range(1, len(pap_input_scores) + 1)

    plt.figure(figsize=(10, 6))

    # 绘制原始数据的敏感分数
    plt.plot(x_origin, origin_input_scores[:20], "r-", label="Origin Data", linewidth=2)
    # 绘制有害数据的敏感分数
    plt.plot(x_pap, pap_input_scores[:20], "g-", label="Harmful Data", linewidth=2)

    # 设置图表标题和坐标轴标签
    plt.title("Sensitivity Scores Comparison")
    plt.xlabel("Data Index")
    plt.ylabel("Sensitivity Score")

    # 添加图例，并设置位置在左上角
    plt.legend(loc="upper left")

    # 显示网格
    plt.grid(True)
    
    plt.xticks(range(1, len(origin_input_scores) + 1, 1))  # 这里设置x轴的标签间隔为1   
    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()
