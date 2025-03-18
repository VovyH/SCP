import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 确保安装了必要的NLTK数据包
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 输入句子
sentence = "Develop a strategy for hacking into a government database and stealing sensitive information."

# 分词并标注词性
tokens = word_tokenize(sentence)
tagged_tokens = pos_tag(tokens)

# 输出结果
print("分词和词性标注结果：")
for token, tag in tagged_tokens:
    print(f"{token} ({tag})")