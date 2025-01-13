import os
import pandas as pd
from tqdm import tqdm
from utils.word_dict.harmful_classfication_utils import harmful_classification
from utils.data_util.data_utils import data_reader
from config import config_args
from utils.word_dict.harmful_classfication_utils import test_prefixes
from utils.word_dict.detect_safety_problem import positive_prefixes

test_model = config_args.test_model
judge_model = config_args.judge_model

def save_data_on_error(data_list, output_file_path, error_msg):
    """
    在发生错误时保存数据到CSV文件。
    """
    print(f"Error occurred: {error_msg}")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    output_df = pd.DataFrame(data_list)
    output_df.to_csv(output_file_path, index=False)
    print(f"Data has been saved to {output_file_path} due to an error.")

def evaluate_harmful_content(
    evaluate_file_path,
    origin_harmful_path,
    output_file_path,
    test_prefixes,
):
    """
    对模型输出进行恶意内容评估。
    """
    try:
        # 加载 CSV 文件
        df = pd.read_csv(evaluate_file_path)
        harmful_behaviors = data_reader(origin_harmful_path)

        # 确保存在 output 列
        if "output" not in df.columns:
            raise ValueError("The CSV file does not contain an 'output' column.")

        # 初始化结果列表
        data_list = []
        count = 0  # 恶意内容数量
        total = 0  # 总评估数量

        # 遍历每个输出并进行评估
        for (idx, row), harmful_input in tqdm(
            zip(df.iterrows(), harmful_behaviors), total=len(df), desc="Evaluating Outputs"
        ):
            total += 1
            model_output = row["output"]
            model_input = row["input"]

            # 评估逻辑
            if model_output != "":
                label = harmful_classification(config_args, model_output)
                if "1" in label:  # 标签包含 1 表示恶意内容
                    label = "1"
                    count += 1
                else:
                    label = "0"  # 善意内容
            else:
                label = "0"  # 如果输出为空或包含拒绝前缀，则标记为 0

            # 打印当前处理状态及详细信息
            print(
                "\n################################\n\n"
                f"***** Current Data: {idx+1}/{len(df)} *****\n\n"
                f"***** Original Harmful Input: {harmful_input} *****\n\n"
                f"***** Model Input: {row['input']} *****\n\n"
                f"***** Response of LLM *****\n\n{model_output}\n\n"
                f"***** Test Model: {test_model} *****\n\n"
                f"***** Judge Model: {judge_model} *****\n\n"
                f"***** ASR Label: {label} *****\n\n"
                f"***** Successful: {count} / {total}, ASR: {count / total:.2%} *****\n\n"
                "################################\n"
            )

            # 构建新的数据项并添加到结果
            item_new = {
                "idx": idx,
                "original_harm_behavior": harmful_input,  # 原始行为
                "nested_prompt": model_input,  # 如果需要，您可以指定 prompt（否则留空）
                "test_model": config_args.test_model,
                "model_output": model_output,
                "label": label,  # 标签（0：善意，1：恶意）
            }
            data_list.append(item_new)

        # 计算恶意内容占比
        asr = count / total if total > 0 else 0
        print(f"\n***** Evaluation Completed *****\n")
        print(f"Total Responses Evaluated: {total}")
        print(f"Number of Harmful Responses: {count}")
        print(f"ASR (Attack Success Rate): {asr:.2%}\n")

        # 保存评估结果为 CSV 文件
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        output_df = pd.DataFrame(data_list)
        output_df.to_csv(output_file_path, index=False)

        print(
            f"The evaluation results have been saved to: {os.path.abspath(output_file_path)}"
        )

    except Exception as e:
        save_data_on_error(data_list, output_file_path, str(e))

def evaluate_prefixes(
    evaluate_file_path, origin_harmful_path, output_file_path, test_prefixes
):
    try:
        # 加载 CSV 文件
        df = pd.read_csv(evaluate_file_path)
        harmful_behaviors = data_reader(origin_harmful_path)
        # ... 省略其他代码 ...

        # 保存评估结果为 CSV 文件
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        output_df = pd.DataFrame(data_list)
        output_df.to_csv(output_file_path, index=False)

    except Exception as e:
        save_data_on_error(data_list, output_file_path, str(e))

if __name__ == "__main__":
    # 文件路径配置
    origin_harmful_path = r"E:/code/Chemotherapy/data/harmful_behaviors.csv"
    evaluate_file_path = r"E:/code/Chemotherapy/log/results/advBench_model_responses_gpt3.5.csv"
    output_file_path = r"E:/code/Chemotherapy/log/results/advBench_evaluate_gpt3.5.csv"

    # 调用评估函数
    evaluate_harmful_content(
        evaluate_file_path=evaluate_file_path,
        origin_harmful_path=origin_harmful_path,
        output_file_path=output_file_path,
        test_prefixes=positive_prefixes,
    )