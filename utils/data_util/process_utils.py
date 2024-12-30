import csv

def extract_behaviors_from_csv(input_file_path, output_file_path):
    behaviors = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > 0:
                behaviors.append(row[0])

    # 将提取的行为写入新的CSV文件
    with open(output_file_path, 'w', encoding='utf-8', newline='') as new_file:
        csv_writer = csv.writer(new_file)
        for behavior in behaviors:
            csv_writer.writerow([behavior])

# 假设输入CSV文件的路径是"HarmBench_standard.csv"
input_file_path = "E:/code/Chemotherapy/data/HarmBench_standard.csv"
# 假设输出CSV文件的路径是"behaviors_extracted.csv"
output_file_path = "E:/code/Chemotherapy/data/behaviors_extracted.csv"

# 调用函数并传入输入和输出文件的路径
extract_behaviors_from_csv(input_file_path, output_file_path)

# 打印提取的行为，如果你需要的话
# 由于行为可能很多，这里注释掉，以免输出太长
# with open(output_file_path, 'r', encoding='utf-8') as new_file:
#     for line in new_file:
#         print(line.strip())