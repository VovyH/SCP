import os
import json
from utils.model_utils.llm_responses_utils import get_response,refine_note
from utils.data_util.data_utils import data_reader
from utils.word_dict.role_task_dict import role, task
from utils.word_dict.drug_init_prompt import role_context, note_context
from config import config_args
import pandas as pd
from utils.data_util.process_response import extract_model_response_content
from tqdm import tqdm
import time
from words_negative.drug_input import TokenManager
from utils.word_dict.harmful_classfication_utils import test_prefixes
from utils.word_dict.detect_safety_problem import negative_prefixes
import random

# 1.必要的输入参数
role_context = role_context.replace("{{## Role ##}}", role)
role_context = role_context.replace("{{## Task ##}}", task)
drug_prompt = role_context + "\n" + note_context  # prompt

responses = []  # 用于存储所有响应
data_list = []  # 初始化存储结果的数据列表
mode_success_counts = { # 初始化一个字典来记录expand、rephrase、shorten的成功次数
    "expand": 0,
    "rephrase": 0,
    "shorten": 0
}

output_file_path = f"E:/code/Chemotherapy/log/results/advBench_model_responses_{config_args.test_model}.csv"

# 2.读取数据
data_path = r"E:/code/Chemotherapy/data/harmful_behaviors.csv"
processed_avoid_positive_path = r"E:/code/Chemotherapy/data/positive_input_deepseek.csv"
harmful_behaviors = data_reader(data_path)

# 1.初始化 TokenManager
token_manager = TokenManager()

positive_data = pd.read_csv(processed_avoid_positive_path)
positive_sentences = []
for index, row in tqdm(positive_data.iterrows(), total=len(positive_data), desc="Gaining Positive Input"):
    positive_sentence = row['positive_sentence']
    positive_sentences.append(positive_sentence)

# 2.得到所有的恶意输入 drug_inputs
drug_inputs = [
    drug_prompt.replace("{{## Question ##}}", behavior)
    for behavior in positive_sentences
]

# 最大重试次数和间隔时间
MAX_RETRIES = 5
RETRY_INTERVAL = 30  # 秒

# 3.定义函数来选择模式
def choose_mode():
    # 如果每个模式的成功次数相同，则随机选择一个
    if mode_success_counts["expand"] == mode_success_counts["rephrase"] == mode_success_counts["shorten"]:
        return random.choice(["expand", "rephrase", "shorten"])
    # 否则，选择成功次数最多的模式
    return max(mode_success_counts, key=mode_success_counts.get)

# 4.获取模型响应
for idx, (drug_input, behavior) in enumerate(tqdm(zip(drug_inputs, positive_sentences), desc="get harmful response")):
    retries = 0
    model_output = None
    final_output = None
    mode = None

    # 重试机制
    while retries < MAX_RETRIES:
        try:
            model_output = get_response(config_args.test_model, config_args, drug_input)
            final_output = extract_model_response_content(model_output)
            if final_output is None:
                final_output = model_output
                
            # 重新得到 note 内容进行组合
            if any(model_output.startswith(prefix) for prefix in test_prefixes):
                mode = choose_mode()
                new_note_context = refine_note(config_args.refine_model,mode,note_context,model_output,config_args)
                drug_prompt = role_context + "\n" + new_note_context  # prompt
                drug_input = drug_prompt.replace("{{## Question ##}}", behavior)
                model_output = get_response(config_args.test_model, config_args, drug_input)
                final_output = extract_model_response_content(model_output)
                if final_output is None:
                    final_output = model_output
                    
                # 检查新的 final_output 是否包含负面前缀
                if not any(final_output.startswith(prefix) for prefix in negative_prefixes):
                    # 如果新的 final_output 不包含负面前缀，则增加选择模式的成功次数
                    mode_success_counts[mode] += 1  # 增加选择模式的成功次数
                    
            # 如果响应不是 None，跳出重试循环
            if model_output is not None:
                break
        except Exception as e:
            # 在发生错误时提前保存已有的结果
            output_df = pd.DataFrame(data_list)
            output_df.to_csv(output_file_path, index=False)
            print(f"Data has been saved to: {output_file_path}")
            print(f"Error while getting response: {e}")

        # 如果响应为 None 或发生异常，等待并重试
        retries += 1
        print(f"Retrying {retries}/{MAX_RETRIES} after {RETRY_INTERVAL} seconds...")
        time.sleep(RETRY_INTERVAL)

    # 如果超过最大重试次数，仍未获取有效响应
    if model_output is None:
        print(f"Failed to get a valid response after {MAX_RETRIES} retries.")
        model_output = "No response after retries"

    # 记录输入和输出
    data_list.append({
        "input": drug_input,
        "output": model_output,
        "final_output": final_output
    })

# 5.结果存储到 CSV 文件中
output_df = pd.DataFrame(data_list)
output_file_path = r"E:/code/Chemotherapy/log/results/advBench_model_responses_deepseek_R1.csv"

# 确保目录存在
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 写入到 CSV 文件
output_df.to_csv(output_file_path, index=False)

print(f"\nThe response file has been saved to:\n{output_file_path}\n")
