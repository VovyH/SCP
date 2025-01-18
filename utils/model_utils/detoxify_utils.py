import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import requests
import time
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# 定义 wordnet 和 punkt 数据的存放路径
wordnet_path = os.path.join(nltk.data.path[0], 'corpora', 'wordnet')
punkt_path = os.path.join(nltk.data.path[0], 'tokenizers', 'punkt')

# 检查是否已经下载必要的 NLTK 资源
if not os.path.exists(wordnet_path):
    nltk.download('wordnet')
if not os.path.exists(punkt_path):
    nltk.download('punkt')

# 获取指定单词的近义词
def get_synonyms(word):
    """获取一个词的近义词"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# 获取句子的嵌入
def get_embedding(sentence):
    """获取句子的嵌入向量，支持最多5次重试，每次间隔30秒

    Args:
        sentence (str): 输入句子

    Raises:
        Exception: 如果重试5次仍然失败，则抛出异常

    Returns:
        list: 嵌入向量
    """
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
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 获取与 "avoid" 最相似的 5 个词汇
def get_most_similar_synonyms(word, num_similar=5):
    """获取与给定词最相似的近义词"""
    # 获取近义词
    synonyms = get_synonyms(word)
    
    # 获取目标词的嵌入
    word_embedding = get_embedding(word)
    time.sleep(20)  # 等待以避免API速率限制
    
    # 计算每个近义词与目标词的余弦相似度
    similarity_scores = {}
    for synonym in synonyms:
        if synonym != word:  # 排除原始词本身
            synonym_embedding = get_embedding(synonym)
            time.sleep(20)  # 防止请求过于频繁
            p = synonym_embedding / np.sum(synonym_embedding)
            q = word_embedding / np.sum(word_embedding)
            kl_div = kl_divergence(p, q)
            # cos_sim = cosine_similarity(word_embedding, synonym_embedding)
            similarity_scores[synonym] = kl_div
    
    # 排序并选择最相似的几个词
    sorted_synonyms = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 选择前 num_similar 个相似词
    most_similar_synonyms = [synonym for synonym, _ in sorted_synonyms[:num_similar]]
    
    return most_similar_synonyms

# 对近义词进行情感分类
def classify_sentiment(word):
    """根据情感得分分类单词为负面或正面"""
    analyzer = SentimentIntensityAnalyzer()
    
    # 获取情感分析得分
    sentiment_score = analyzer.polarity_scores(word)['compound']
    
    return sentiment_score

# 获取 "avoid" 的近义词并选择最相似的 5 个词汇
def get_similar_words_and_sentiment(word):
    # 获取与给定词最相似的 5 个近义词
    most_similar_synonyms = get_most_similar_synonyms(word, num_similar=5)
    print(f"与 '{word}' 最相似的 5 个近义词是: {most_similar_synonyms}")
    
    # 对这些词汇进行情感分类
    sentiment_results = {}
    for synonym in most_similar_synonyms:
        sentiment_results[synonym] = classify_sentiment(synonym)
    
    return sentiment_results

# 主函数：结合获取近义词的相似度和情感分析
def evaluate_similarity_and_safety(word):
    """评估与目标词最相似的近义词及其情感分析"""
    sentiment_results = get_similar_words_and_sentiment(word)
    
    # 输出情感分析结果
    print("情感得分结果：")
    for synonym, sentiment in sentiment_results.items():
        print(f"近义词: {synonym}, 情感得分: {sentiment}")

# 执行程序
evaluate_similarity_and_safety("avoid")
