import re


def extract_model_response_content(text):
    # 使用正则表达式匹配 "Final step:" 后的所有内容
    pattern_step = r"Final step:\s*(.*?)(?=\n\n|$)"
    pattern_example = r"Example:\s*(.*?)(?=\n\n|$)"
    match_step = re.search(pattern_step, text, re.DOTALL)
    match_example = re.search(pattern_example, text, re.DOTALL)

    if match_step and match_step:
        # 返回匹配到的内容，并去除多余的空白字符
        return match_step.group(1).strip() + "\n" + match_example.group(1).strip()
    else:
        return None



