import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import requests
import time
from tqdm import tqdm

def get_sentence_similarity(sentence1, sentence2):
    """计算s1和s2之间的差异

    Args:
        sentence1 (_type_): s1
        sentence2 (_type_): s2

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    # 计算句子的向量
    def get_embedding(sentence):
        """获取句子的嵌入向量，支持最多5次重试，每次间隔30秒

        Args:
            sentence (str): 输入句子

        Raises:
            Exception: 如果重试5次仍然失败，则抛出异常

        Returns:
            list: 嵌入向量
        """
        # 嵌入 API URL
        url = "https://api.siliconflow.cn/v1/embeddings"  # 替换为实际的嵌入服务地址
        payload = {
            "model": "BAAI/bge-m3",
            "input": sentence
        }
        headers = {
            "Authorization": "Bearer sk-kifqepgmrmlstabhxlmxrylkppvcppumtvwcdbwajgotuvvk",
            "Content-Type": "application/json"
        }
    
        max_retries = 5  # 最大重试次数
        for attempt in range(max_retries):
            try:
                # 调用 API 获取嵌入向量
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    # 返回嵌入向量
                    return response.json()['data'][0]['embedding']
                else:
                    # 错误处理，抛出异常
                    raise Exception(f"Failed to get embedding: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30)  # 如果未达到最大尝试次数，则等待30秒后重试
                else:
                    print("Max retries reached. Unable to get embedding.")
                    raise e  # 如果达到最大重试次数，抛出异常

    # 计算KL散度
    def kl_divergence(p, q):
        # 计算KL散度，确保没有零概率
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return entropy(p, q)

    # 计算余弦相似度
    def cosine_similarity(vec1, vec2):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    # 获取句子的嵌入
    embedding1 = get_embedding(sentence1)
    time.sleep(20)  # 等待以避免API速率限制
    embedding2 = get_embedding(sentence2)

    # 将嵌入向量归一化为概率分布
    p = embedding1 / np.sum(embedding1)
    q = embedding2 / np.sum(embedding2)

    # 计算KL散度
    kl_div = kl_divergence(p, q)

    # 计算余弦相似度
    cos_sim = cosine_similarity(embedding1, embedding2)

    # 综合两种方法，返回一个相似性度量 (你可以根据需求调整权重)
    similarity_score = {
        "cosine_similarity": cos_sim,
        "kl_divergence": kl_div,
        "final_similarity_score": (cos_sim - kl_div)  # 这里的计算方式可以根据需求调整
    }

    return similarity_score

def compute_similarity_for_dataset(input_file, output_file):
    data = pd.read_csv(input_file)
    
    similarity_results = []
    
    # 使用 tqdm 显示进度条
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Calculating Similarity"):
        original_sentence = row['original_sentence']
        positive_sentence = row['positive_sentence']
        
        try:
            similarity = get_sentence_similarity(original_sentence, positive_sentence)
        except Exception as e:
            print(f"Error at index {index}: {e}")
            continue
        
        similarity_results.append({
            "original_sentence": original_sentence,
            "positive_sentence": positive_sentence,
            "cosine_similarity": similarity["cosine_similarity"],
            "kl_divergence": similarity["kl_divergence"],
            "final_similarity_score": similarity["final_similarity_score"]
        })
    
    similarity_df = pd.DataFrame(similarity_results)
    
    similarity_df.to_csv(output_file, index=False)
    print(f"Similarity results saved to {output_file}")

# 输入文件和输出文件路径
input_file = 'E:/code/Chemotherapy/data/gpt_positive_avoid_results22.csv'
output_file = 'E:/code/Chemotherapy/data/behaviors_similarity_results_normal.csv'

# 执行函数
compute_similarity_for_dataset(input_file, output_file)
