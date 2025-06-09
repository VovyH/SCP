import os
import pandas as pd
from tqdm import tqdm
from utils.word_dict.harmful_classfication_utils import harmful_classification
from utils.data_util.data_utils import data_reader
import os
from config import config_args
from utils.word_dict.harmful_classfication_utils import test_prefixes
from utils.word_dict.detect_safety_problem import negative_prefixes
import json
import concurrent.futures
from queue import Queue
import threading

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
    """
    评估 CSV 文件中每行的输出是否包含指定的前缀列表。

    参数:
        evaluate_file_path (str): CSV 文件路径
        origin_harmful_path:原恶意内容路径
        test_prefixes (list): 包含需要检查的前缀列表

    返回:
        tuple: 包含不包含指定前缀的行数和比例
    """

    # 加载 CSV 文件
    df = pd.read_csv(evaluate_file_path)
    harmful_behaviors = data_reader(origin_harmful_path)

    # 确保存在 'output' 列
    if "output" not in df.columns:
        raise ValueError("The CSV file does not contain an 'output' column.")

    # 初始化计数器
    count = 0  # 不包含指定前缀的行数
    total = 0  # 总行数
    data_list = []  # 存储每行的处理结果

    # 遍历每行输出
    for (idx, row), harmful_input in tqdm(
        zip(df.iterrows(), harmful_behaviors), total=len(df), desc="Evaluating Outputs"
    ):
        model_input = row["input"]
        model_output = row["output"]
        total += 1

        if not any(model_output.startswith(prefix) for prefix in test_prefixes):
            count += 1
            label = "1"  # 标签为"无前缀"
        else:
            label = "0"  # 标签为"包含前缀"

        # 打印每条数据处理状态
        print(
            "\n################################\n\n"
            f"***** Current Data: {idx + 1}/{len(df)} *****\n\n"
            f"***** Original Harmful Input: {harmful_input} *****\n\n"
            f"***** Model Input: {model_input} *****\n\n"
            f"***** Response of LLM *****\n\n{model_output}\n\n"
            f"***** Test Model: {test_model} *****\n\n"
            f"***** ASR Label: {label} *****\n\n"
            f"***** Successful: {count} / {total}, Prefix Absence Rate: {count / total:.2%} *****\n\n"
            "################################\n"
        )

        # 构建新的数据项并添加到结果
        item_new = {
            "idx": idx,
            "original_harm_behavior": harmful_input,
            "nested_prompt": model_input,
            "model_output": model_output,
            "ASR Label": label,
        }
        data_list.append(item_new)

    # 计算总行数和比例
    # 保存评估结果为 CSV 文件
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 写入到 CSV 文件
    output_df = pd.DataFrame(data_list)
    output_df.to_csv(output_file_path, index=False)
    
    proportion = count / total if total > 0 else 0

    # 打印最终统计结果
    print(f"\n***** Evaluation Completed *****\n")
    print(f"Total Responses Evaluated: {total}")
    print(f"Number of Harmful Responses: {count}")
    print(f"Prefix Absence Rate: {proportion:.2%}\n")

    # return count, proportion, data_list


def process_batch(batch_data, config_args):
    """
    处理一批数据的函数
    """
    results = []
    count = 0
    total = len(batch_data)
    
    for idx, (item, harmful_input) in enumerate(batch_data):
        content = item.get('content', item.get('response', item.get('output', '')))
        if not content and 'message' in item:
            content = item['message'].get('content', '')
            
        if not content:
            continue
            
        model_input = item.get('input', item.get('prompt', f"未知输入 #{idx}"))
        
        # 评估逻辑
        if content != "":
            label = harmful_classification(config_args, content)
            if "1" in label:
                label = "1"
                count += 1
            else:
                label = "0"
        else:
            label = "0"
            
        results.append({
            "idx": idx,
            "original_harm_behavior": harmful_input,
            "nested_prompt": model_input,
            "model_output": content,
            "ASR Label": label,
        })
        
    return results, count


def evaluate_json_content_parallel(
    json_file_path, 
    origin_harmful_path, 
    output_file_path, 
    test_prefixes,
    batch_size=10,
    max_workers=5
):
    """
    并发处理JSON文件内容的评估函数
    """
    try:
        # 加载数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        harmful_behaviors = data_reader(origin_harmful_path)
        
        # 准备数据批次
        batches = []
        current_batch = []
        
        for idx, item in enumerate(json_data):
            harmful_input = harmful_behaviors[idx] if idx < len(harmful_behaviors) else f"未知输入 #{idx}"
            current_batch.append((item, harmful_input))
            
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []
                
        if current_batch:  # 添加最后一个不完整的批次
            batches.append(current_batch)
            
        # 结果收集
        all_results = []
        total_count = 0
        total_processed = 0
        
        # 进度条
        pbar = tqdm(total=len(json_data), desc="Processing batches")
        
        # 使用线程池处理批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次的任务
            future_to_batch = {
                executor.submit(process_batch, batch, config_args): batch 
                for batch in batches
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    results, count = future.result()
                    all_results.extend(results)
                    total_count += count
                    total_processed += len(results)
                    pbar.update(len(results))
                    
                    # 定期保存中间结果
                    if len(all_results) % (batch_size * 2) == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_path = output_file_path.replace('.csv', f'_temp_{len(all_results)}.csv')
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        temp_df.to_csv(temp_path, index=False)
                        print(f"\n临时保存数据到: {temp_path}")
                        
                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    continue
        
        pbar.close()
        
        # 保存最终结果
        proportion = total_count / total_processed if total_processed > 0 else 0
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_file_path, index=False)
        
        # 打印统计结果
        print(f"\n***** Evaluation Completed *****")
        print(f"Total Responses Evaluated: {total_processed}")
        print(f"Number of Harmful Responses: {total_count}")
        print(f"Harmful Rate: {proportion:.2%}\n")
        
    except Exception as e:
        error_path = output_file_path.replace('.csv', '_error.csv')
        save_data_on_error(all_results, error_path, str(e))
        raise


def evaluate_json_content_parallel_continue(
    json_file_path, 
    origin_harmful_path, 
    output_file_path, 
    test_prefixes,
    start_idx=453,  # 从第453条开始
    batch_size=10,
    max_workers=5
):
    """
    从指定索引继续并发处理JSON文件内容的评估函数
    """
    try:
        # 加载数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        harmful_behaviors = data_reader(origin_harmful_path)
        
        # 只处理从start_idx开始的数据
        json_data = json_data[start_idx:]
        harmful_behaviors = harmful_behaviors[start_idx:]
        
        # 准备数据批次
        batches = []
        current_batch = []
        
        for idx, item in enumerate(json_data):
            # 保持原始索引，方便后续对应
            real_idx = idx + start_idx
            harmful_input = harmful_behaviors[idx] if idx < len(harmful_behaviors) else f"未知输入 #{real_idx}"
            current_batch.append((item, harmful_input, real_idx))
            
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []
                
        if current_batch:  # 添加最后一个不完整的批次
            batches.append(current_batch)
            
        # 结果收集
        all_results = []
        total_count = 0
        total_processed = 0
        
        # 进度条
        pbar = tqdm(total=len(json_data), desc=f"Processing batches from index {start_idx}")
        
        # 使用线程池处理批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch_with_real_idx, batch, config_args): batch 
                for batch in batches
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    results, count = future.result()
                    all_results.extend(results)
                    total_count += count
                    total_processed += len(results)
                    pbar.update(len(results))
                    
                    # 定期保存中间结果
                    if len(all_results) % (batch_size * 2) == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_path = output_file_path.replace('.csv', f'_continue_{start_idx}_temp_{len(all_results)}.csv')
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        temp_df.to_csv(temp_path, index=False)
                        print(f"\n临时保存数据到: {temp_path}")
                        
                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    continue
        
        pbar.close()
        
        # 保存最终结果
        proportion = total_count / total_processed if total_processed > 0 else 0
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        final_output_path = output_file_path.replace('.csv', f'_continue_from_{start_idx}.csv')
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(final_output_path, index=False)
        
        # 打印统计结果
        print(f"\n***** Evaluation Completed *****")
        print(f"Total Responses Evaluated: {total_processed}")
        print(f"Number of Harmful Responses: {total_count}")
        print(f"Harmful Rate: {proportion:.2%}\n")
        print(f"Results saved to: {final_output_path}")
        
    except Exception as e:
        error_path = output_file_path.replace('.csv', f'_continue_{start_idx}_error.csv')
        save_data_on_error(all_results, error_path, str(e))
        raise


def process_batch_with_real_idx(batch_data, config_args):
    """
    处理一批数据的函数（带有真实索引）
    """
    results = []
    count = 0
    
    for item, harmful_input, real_idx in batch_data:
        content = item.get('content', item.get('response', item.get('output', '')))
        if not content and 'message' in item:
            content = item['message'].get('content', '')
            
        if not content:
            continue
            
        model_input = item.get('input', item.get('prompt', f"未知输入 #{real_idx}"))
        
        # 评估逻辑
        if content != "":
            label = harmful_classification(config_args, content)
            if "1" in label:
                label = "1"
                count += 1
            else:
                label = "0"
        else:
            label = "0"
            
        results.append({
            "idx": real_idx,  # 使用真实索引
            "original_harm_behavior": harmful_input,
            "nested_prompt": model_input,
            "model_output": content,
            "ASR Label": label,
        })
        
    return results, count


if __name__ == "__main__":
    # 文件路径配置
    origin_harmful_path = r"E:/我的论文和代码/Chemotherapy/data/harmful_behaviors.csv"
    evaluate_file_path = r"E:\我的论文和代码\Chemotherapy\log\results\subset_gpt-4-0613.csv"
    json_output_path = f"E:/我的论文和代码/Chemotherapy/log/results/json_evaluate_deepseek_111111.csv"
    evaluate_defense_path = r"E:\我的论文和代码\Chemotherapy\log\results\processed_responses.csv"
    evaluate_defense_output_path = r"E:\我的论文和代码\Chemotherapy\log\results\subset_gpt-4-0613_evaluate.csv.csv"
    
    # 从第453条开始继续处理
    evaluate_prefixes(
        evaluate_file_path=evaluate_defense_path,
        origin_harmful_path=origin_harmful_path,
        output_file_path=evaluate_defense_output_path,
        test_prefixes=test_prefixes
    )
