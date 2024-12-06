import nltk
from nltk.corpus import wordnet as wn

# 1.定义一个函数来获取传入动词的反义词
def get_antonyms(verb):
    """获取 verb 的反义词

    Args:
        verb (_type_): 动词词汇

    Returns:
        _type_: 反义词集合
    """
    antonyms = set()  # 创建一个集合来存储反义词，避免重复
    
    # 1.获取所有与输入动词相关的同义词集（synsets），并指定词性为动词（VERB）
    synsets = wn.synsets(verb, pos=wn.VERB)
    
    # 2.首先尝试获取反义词
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    
    # 3.如果没有找到反义词，则开始获取近义词及其扩展近义词
    if not antonyms:
        synonyms_to_process = set()
        
        # 3.1 第一轮：获取直接近义词，并过滤掉包含下划线的词汇
        for syn in synsets:
            for lemma in syn.lemmas():
                if '_' not in lemma.name():
                    synonyms_to_process.add(lemma.name())
        
        # 3.2 进行两轮扩展近义词，并过滤掉包含下划线的词汇
        for _ in range(2):
            new_synonyms = set()
            for synonym in synonyms_to_process:
                for syn in wn.synsets(synonym, pos=wn.VERB):
                    for lemma in syn.lemmas():
                        if '_' not in lemma.name():
                            new_synonyms.add(lemma.name())
            synonyms_to_process.update(new_synonyms)
        
        # 3.3 将最终的近义词置反，并过滤掉包含下划线的词汇
        for synonym in synonyms_to_process:
            for syn in wn.synsets(synonym, pos=wn.VERB):
                for lemma in syn.lemmas():
                    if lemma.antonyms() and '_' not in lemma.antonyms()[0].name():
                        antonyms.add(lemma.antonyms()[0].name())
    
    return list(antonyms)  # 返回反义词列表

# 定义一个函数来获取传入动词的上位词汇
def get_hypernyms(verb):
    hypernyms = set()  # 创建一个集合来存储上位词汇，避免重复
    # 获取所有与输入动词相关的同义词集（synsets），并指定词性为动词（VERB）
    for syn in wn.synsets(verb, pos=wn.VERB):
        # 遍历每个同义词集的上位词汇
        for hypernym in syn.hypernyms():
            # 遍历每个上位词汇中的词条（lemmas）
            for lemma in hypernym.lemmas():
                # 将上位词汇的词条名称添加到集合中，如果名称不包含下划线
                if '_' not in lemma.name():
                    hypernyms.add(lemma.name())
    return list(hypernyms)  # 返回上位词汇列表

# 示例用法
if __name__ == "__main__":
    verb = "write"  # 定义要查询的动词
    # 打印该动词的反义词
    print(f"Antonyms of '{verb}': {get_antonyms(verb)}")
    # 打印该动词的上位词汇
    print(f"Hypernyms of '{verb}': {get_hypernyms(verb)}")



