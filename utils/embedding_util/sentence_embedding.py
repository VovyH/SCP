import requests
from numpy import dot
from numpy.linalg import norm

# def get_sentence_similarity(sentence1, sentence2):
#     def get_embedding(sentence):
#         url = "https://api.siliconflow.cn/v1/embeddings"
#         payload = {
#             "model": "BAAI/bge-m3",
#             "input": sentence
#         }
#         headers = {
#             "Authorization": "Bearer sk-kifqepgmrmlstabhxlmxrylkppvcppumtvwcdbwajgotuvvk",
#             "Content-Type": "application/json"
#         }
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json()['data'][0]['embedding']
#         else:
#             raise Exception(f"Failed to get embedding: {response.text}")

#     # 计算余弦相似度
#     def cosine_similarity(vec1, vec2):
#         return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

#     # 获取句子的嵌入
#     embedding1 = get_embedding(sentence1)
#     embedding2 = get_embedding(sentence2)

#     # 计算并返回相似度
#     similarity = cosine_similarity(embedding1, embedding2)
#     return similarity


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
        sentence1 (str): 第一句
        sentence2 (str): 第二句

    Returns:
        dict: 包含余弦相似度和KL散度的字典
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
        """计算KL散度，确保没有零概率"""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return entropy(p, q)

    # 计算余弦相似度
    def cosine_similarity(vec1, vec2):
        """计算两个向量的余弦相似度"""
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

# 示例调用
sentence1 = "avoid"
sentence2 = "avert"

similarity_score = get_sentence_similarity(sentence1, sentence2)
# print(f"句子 '{sentence1}' 和句子 '{sentence2}' 的相似度为: {similarity_score:.4f}")
print(similarity_score)