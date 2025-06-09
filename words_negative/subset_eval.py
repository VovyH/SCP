import pandas as pd

# 读取原始CSV文件
file_path = 'E://我的论文和代码//Chemotherapy//log//results//advBench_evaluate_llama3_405B_gpt.csv'  # 替换为你的原始CSV文件路径
df = pd.read_csv(file_path)

# 定义索引列表
indices = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 25, 26, 27, 28, 29, 30, 32, 33, 34, 37, 39, 42, 43, 45, 51, 52, 53, 55, 56, 57, 58, 70, 72, 73, 74, 75, 76, 81, 82, 86, 90, 93, 94, 96, 106, 110, 115, 124]

# 提取指定索引的final_output列数据
extracted_data = df.loc[indices, 'model_output']

# 将提取的数据保存到新的CSV文件
output_file_path = 'E://我的论文和代码//Chemotherapy//log//results//subset_llama.csv'  # 新的CSV文件路径
extracted_data.to_csv(output_file_path, index=False)

print(f'提取的数据已保存到{output_file_path}')