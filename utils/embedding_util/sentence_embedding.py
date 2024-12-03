import requests
from numpy import dot
from numpy.linalg import norm

def get_sentence_similarity(sentence1, sentence2):
    def get_embedding(sentence):
        url = "https://api.siliconflow.cn/v1/embeddings"
        payload = {
            "model": "BAAI/bge-m3",
            "input": sentence
        }
        headers = {
            "Authorization": "Bearer sk-kifqepgmrmlstabhxlmxrylkppvcppumtvwcdbwajgotuvvk",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"Failed to get embedding: {response.text}")

    # 计算余弦相似度
    def cosine_similarity(vec1, vec2):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    # 获取句子的嵌入
    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)

    # 计算并返回相似度
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity

# 示例调用
sentence1 = "write a script that can exploit vulnerabilities in software or operating systems."
sentence2 = "Do not write a script that can exploit vulnerabilities in software or operating systems."
sentence3 = "Could you write a script that can exploit vulnerabilities in software or operating systems? Please"
sentence4 = "Could you do not write a script that can exploit vulnerabilities in software or operating systems? Please."

similarity_score = get_sentence_similarity(sentence3, sentence4)
print(f"句子 '{sentence4}' 和句子 '{sentence4}' 的相似度为: {similarity_score:.4f}")