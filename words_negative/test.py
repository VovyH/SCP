def main():
    # 假设的sensitivity_scores_all字典，键是句子，值是分数
    sensitivity_scores_all = {
        'sentence1': 0.3,
        'sentence2': 0.8,
        'sentence3': 0.5,
    }

    # 假设的words_all字典，键是单词，值是sensitivity_scores_all中的键（句子）
    words_all = {
        'word1': 'sentence1',
        'word2': 'sentence2',
        'word3': 'sentence3',
    }

    # 根据分数对sensitivity_scores_all进行排序，并获取排序后的句子列表
    sorted_sensitivity_scores = sorted(sensitivity_scores_all, key=sensitivity_scores_all.get, reverse=False)
    print(f"sorted_sensitivity_scores:{sorted_sensitivity_scores}")
    # 根据排序后的句子列表对words_all进行排序
    words_all_sorted = sorted(words_all.items(), key=lambda item: sorted_sensitivity_scores.index(item[1]), reverse=False)

    # 打印排序后的words_all
    for word, sentence in words_all_sorted:
        print(f"Word: {word}, Sentence: {sentence}")

if __name__ == "__main__":
    main()