import os
import json
from utils.model_utils.llm_responses_utils import get_response
from utils.data_util.data_utils import data_reader
from utils.word_dict.role_task_dict import role, task
from utils.word_dict.drug_init_prompt import role_context, note_context
from config import config_args
from utils.word_dict.harmful_classfication_utils import (
    test_prefixes,
    harmful_classification,
)
from utils.data_util.process_response import extract_model_response_content
from words_transfer.words_transfer import (
    transform_and_evaluate,
    get_antonyms,
    find_all_verb_indices,
)
from tqdm import tqdm
from words_negative.drug_input import TokenManager

# 1.必要的输入参数
drug_prompt = role_context + "\n" + note_context  # prompt
drug_prompt = drug_prompt.replace("{{## Role ##}}", role)
drug_prompt = drug_prompt.replace("{{## Task ##}}", task)

model_type = config_args.test_model  # model
responses = []  # 用于存储所有响应
count = 0  # 记为恶意内容数量
total = 0  # 输出总数
data_list = []  # 初始化存储结果的数据列表

# 2.读取数据
data_path = r"E:/code/Chemotherapy/data/harmful_behaviors.csv"
harmful_behaviors = data_reader(data_path)

# 初始化 TokenManager
token_manager = TokenManager()
model_type = "gpt-3.5-turbo-0613"
results,positive_sentences = token_manager.process_input_sentences(model_type, harmful_behaviors)

# 3.得到所有的恶意输入 drug_inputs
drug_inputs = [
    drug_prompt.replace("{{## Question ##}}", behavior)
    for behavior in positive_sentences
]

# 4.得到 harmful response
for idx, drug_input in enumerate(tqdm(drug_inputs, desc="get harmful response")):
    total += 1
    # 4.1 获取响应
    model_output = get_response(model_type, config_args, drug_input)
    model_output = extract_model_response_content(model_output)

    # 4.2 进行评估
    if (
        not any([prefix in model_output for prefix in test_prefixes])
        and model_output != ""
    ):
        # 如果输出的内容不包含拒绝前缀prefix，并且输出为空的话，则调用有害分类函数
        label = harmful_classification(config_args, model_output)
        if "1" in label:  # 标签包含1说明是恶意内容
            label = "1"
            count += 1
        else:
            label = "0"  # 为0说明为善意内容
    else:
        label = "0"  # 拒绝回答直接回答或者输出为空，直接标记为0

    # 4.3 存入到结果集合当中
    responses.append(model_output)

    # 4.4打印当前处理状态及详细信息
    print(
        "\n################################\n\n"
        f"***** File Name *****\n{config_args.data_path}\n\n"
        f"***** Original Data: {harmful_behaviors[idx]} *****\n\n"
        f"***** Current Data: {idx+1}/{len(drug_input)} *****\n\n"
        f"***** Response of LLM *****\n\n{model_output}\n\n"
        f"***** Test Model: {config_args.test_model} *****\n\n"
        f"***** Judge Model: {config_args.judge_model} *****\n\n"
        f"***** ASR Label: {label} *****\n\n"
        f"***** Successful: {count} / {total}, ASR: {count / total} *****\n\n"
        "################################\n"
    )

    # 4.5构建新的数据将并添加到结果
    item_new = {}
    item_new["idx"] = idx
    item_new["original_harm_behavior"] = harmful_behaviors[idx]
    item_new["nested_prompt"] = drug_input
    item_new["test_model"] = config_args.test_model
    item_new["model_output"] = model_output

    # 将新构建的数据项添加到结果列表中
    data_list.append(item_new)

# 5.结果存储到JSON文件中
if not os.path.exists("./results/responses"):
    os.makedirs("./results/responses")

file_name = f"./results/responses/responses_of_{config_args.test_model}.json"

with open(file_name, "w") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

file_path = os.path.abspath(file_name)
print(f"\nThe response file has been saved to:\n{file_path}\n")
