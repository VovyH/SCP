# 导入必要的模块
from utils.detect_toxic_utils.sensitive_detect import predict_sensitivity  # 导入敏感性预测函数
from config import config_args  # 导入配置信息
from utils.word_dict.role_different_prompt import user_context, llm_context  # 导入不同的角色 prompt
from utils.data_util.data_utils import data_reader, data_read_advbench_sub  # 读取恶意输入
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random

user_input = user_context
llm_input = llm_context

# 日志目录的创建
log_dir = "E:\code\Chemotherapy\log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 日志文件路径
log_file_path = os.path.join(log_dir, "role.log")

# 主程序
def main():
    # 1.读取数据文件中的每一行
    data = data_reader(config_args.data_path)
    # assert len(data) == 520

    results = []

    for idx, question in enumerate(
        tqdm(data, desc="Evaluating the difference of role in prompt"), start=1
    ):
        question = question.strip()  # 去除换行符

        # 1.替换占位符
        user_context = user_input.replace("{{## Question ##}}", question)
        llm_context = llm_input.replace("{{## Question ##}}", question)

        # 2.预测敏感性分数
        user_sensitive_score = predict_sensitivity(user_context)
        llm_sensitive_score = predict_sensitivity(llm_context)

        # 3.返回同一问题下不同提示的敏感分数
        results.append(
            {
                "index": idx,
                "question": question,
                "user_sensitive_score": user_sensitive_score,
                "llm_sensitive_score": llm_sensitive_score,
            }
        )

    # 打印结果
    with open(log_file_path, "w") as log_file:
        for result in results:
            # 1.打印到控制台上
            print(f"Epoch {result['index']}:")
            print(f"Question: {result['question']}")
            print(
                f"User Context Sensitive Score: {result['user_sensitive_score']:.4f}%"
            )
            print(f"LLM Context Sensitive Score: {result['llm_sensitive_score']:.4f}%")
            print()  # 打印空行以分隔不同问题的结果

            # 2.输出到日志文件中
            log_file.write(f"Epoch {result['index']}:\n")
            log_file.write(f"Question: {result['question']}\n")
            log_file.write(
                f"User Context Sensitive Score: {result['user_sensitive_score']:.4f}%\n"
            )
            log_file.write(
                f"LLM Context Sensitive Score: {result['llm_sensitive_score']:.4f}%\n"
            )
            log_file.write("\n")  # 打印空行以分隔不同问题的结果

    # 绘制图表，随机抽取50条数据进行绘图
    sampled_results = random.sample(results, 50)

    # 提取用户和LLM的敏感分数
    user_scores = [result["user_sensitive_score"] for result in sampled_results]
    llm_scores = [result["llm_sensitive_score"] for result in sampled_results]

    # 计算最小值和最大值
    min_user_score = min(user_scores)
    max_user_score = max(user_scores)
    min_llm_score = min(llm_scores)
    max_llm_score = max(llm_scores)

    # 归一化函数
    def normalize(score, min_score, max_score):
        return 2 * ((score - min_score) / (max_score - min_score)) - 1

    # 归一化敏感分数
    normalized_user_scores = [normalize(score, min_user_score, max_user_score) for score in user_scores]
    normalized_llm_scores = [normalize(score, min_llm_score, max_llm_score) for score in llm_scores]
    show_epochs = list(range(1, len(sampled_results) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(
        show_epochs, normalized_user_scores, label="User Context Sensitive Score", color="green"
    )
    plt.plot(show_epochs, normalized_llm_scores, label="LLM Context Sensitive Score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Sensitive Score (-1 to 1)")
    plt.title("Comparison of User and LLM Context Sensitive Scores")
    plt.legend(loc='upper left')  # 将图例放在左上角
    plt.grid(True)
    plt.ylim(-1.1, 1.1)  # 设置y轴范围，确保所有数据都在范围内
    plt.savefig(os.path.join(log_dir, "sensitive_scores_comparison.png"))
    plt.show()

if __name__ == "__main__":
    main()



