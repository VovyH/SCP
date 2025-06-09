import nltk
from nltk.corpus import wordnet as wn

# 下载必要的 NLTK 数据
nltk.download('wordnet')
nltk.download('omw')

def get_synonyms(word, lang='eng'):
    """
    获取一个单词的所有近义词
    :param word: 要查询的单词
    :param lang: 语言代码，默认为 'eng'（英语）
    :return: 包含所有近义词的集合
    """
    synonyms = set()
    
    # 获取单词的所有同义词集 (synsets)
    synsets = wn.synsets(word, lang=lang)
    
    for syn in synsets:
        # 遍历每个同义词集中的所有 lemma
        for lemma in syn.lemmas(lang=lang):
            synonyms.add(lemma.name())
    
    return synonyms

# 获取单词 "protect" 的所有近义词
word = "protect"
synonyms = get_synonyms(word)

# 打印结果
print(f"近义词列表：{', '.join(synonyms)}")