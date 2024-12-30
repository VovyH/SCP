from pattern3.text.en import conjugate, PRESENT, PROGRESSIVE

def get_present_progressive(verb):
    """
    接受一个英文动词作为输入，返回该动词的现在分词形式。
    
    参数:
    verb (str): 需要转换为现在分词形式的动词
    
    返回:
    str: 动词的现在分词形式
    """
    try:
        present_progressive = conjugate(verb, tense=PRESENT, aspect=PROGRESSIVE)
        return present_progressive
    except Exception as e:
        print(f"An error occurred while conjugating '{verb}': {e}")
        return None

