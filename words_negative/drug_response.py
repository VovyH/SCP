import os
import json
from utils.model_utils.llm_responses_utils import get_response
from utils.data_util.data_utils import data_reader
from utils.word_dict.role_task_dict import role, task
from utils.word_dict.drug_init_prompt import role_context, note_context
from config import config_args
import pandas as pd
from utils.data_util.process_response import extract_model_response_content
from tqdm import tqdm
import time
from words_negative.drug_input import TokenManager

# 1.必要的输入参数
drug_prompt = role_context + "\n" + note_context  # prompt
drug_prompt = drug_prompt.replace("{{## Role ##}}", role)
drug_prompt = drug_prompt.replace("{{## Task ##}}", task)

responses = []  # 用于存储所有响应
data_list = []  # 初始化存储结果的数据列表

# 2.读取数据
data_path = r"E:/code/Chemotherapy/data/harmful_behaviors.csv"
processed_avoid_positive_path = r"E:/code/Chemotherapy/data/gpt_positive_avoid_results.csv"
harmful_behaviors = data_reader(data_path)

# 初始化 TokenManager
token_manager = TokenManager()

positive_data = pd.read_csv(processed_avoid_positive_path)
positive_sentences = []
for index, row in tqdm(positive_data.iterrows(), total=len(positive_data), desc="Gaining Positive Input"):
    positive_sentence = row['positive_sentence']
    positive_sentences.append(positive_sentence)

# 3.得到所有的恶意输入 drug_inputs
drug_inputs = [
    drug_prompt.replace("{{## Question ##}}", behavior)
    for behavior in positive_sentences
]

# 最大重试次数和间隔时间
MAX_RETRIES = 5
RETRY_INTERVAL = 30  # 秒

# 4.获取模型响应
for idx, drug_input in enumerate(tqdm(drug_inputs, desc="get harmful response")):
    retries = 0
    model_output = None

    # 重试机制
    while retries < MAX_RETRIES:
        try:
            model_output = get_response(config_args.test_model, config_args, drug_input)
            final_output = extract_model_response_content(model_output)

            # 如果响应不是 None，跳出重试循环
            if model_output is not None:
                break
        except Exception as e:
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
output_file_path = r"E:/code/Chemotherapy/log/results/model_responses.csv"

# 确保目录存在
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 写入到 CSV 文件
output_df.to_csv(output_file_path, index=False)

print(f"\nThe response file has been saved to:\n{output_file_path}\n")
