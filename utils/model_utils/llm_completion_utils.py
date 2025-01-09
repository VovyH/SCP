import time
from openai import OpenAI  # 导入 OpenAI 客户端库
from config import Config  # 导入 Config 配置类
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT  # 导入 Anthropic 客户端库
import http.client
import json
import re


def chatCompletion(messages, config: Config):
    """调用 OpenAI API 获取聊天补全结果"""
    model = config.test_model
    temperature = config.temperature
    retry_times = config.retry_times
    round_sleep = config.round_sleep
    fail_sleep = config.fail_sleep
    api_key = config.gpt_api_key
    base_url = config.gpt_base_url

    # 初始化 OpenAI 客户端
    client = (
        OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else OpenAI(api_key=api_key)
    )

    response = None  # 初始化 response 变量
    for retry_time in range(retry_times):  # 尝试重试
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            break  # 成功后退出循环
        except Exception as e:
            print(f"Retry {retry_time + 1}/{retry_times} failed for {model}: {e}")
            time.sleep(fail_sleep)  # 重试前等待一段时间

    # 如果所有尝试均失败
    if response is None:
        raise RuntimeError(
            f"Failed to get a response from the model after {retry_times} retries."
        )

    # 成功后解析响应
    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)  # 等待指定的时间间隔

    return model_output


def claudeCompletion(messages, config: Config):
    """
    调用 Claude API 获取讲笑话的响应，并返回处理后的完整笑话。
    带有重试机制，最大重试次数为3，每次间隔30秒。
    """
    max_retries = config.retry_times  # 最大重试次数
    retry_delay = 30  # 每次重试的延迟时间（秒）
    
    for attempt in range(max_retries):
        try:
            # 建立HTTPS连接
            conn = http.client.HTTPSConnection("oa.api2d.net")
            
            # 设置请求参数
            payload = json.dumps(
                {
                    "model": config.test_model,
                    "messages": [{"role": "user", "content": messages}],
                    "stream": True,
                    "max_tokens": config.max_tokens,
                }
            )
            headers = {
                "Authorization": "Bearer fk229908-aQM21ZSHgeNJy2XIDHZmXFDAyl8imzjF",  # 替换为你的实际API密钥
                "Content-Type": "application/json",
            }
            
            # 发起请求
            conn.request("POST", "/claude/v1/messages", payload, headers)
            res = conn.getresponse()

            # 读取响应数据
            data = res.read().decode("utf-8")

            # 提取所有的 text 字段内容
            text_fragments = re.findall(r'"text":"(.*?)"', data)

            # 合并 text 内容
            combined_text = "".join(text_fragments)

            # 返回处理后的内容
            return combined_text

        except Exception as e:
            # 如果不是最后一次尝试，进行延迟重试
            if attempt < max_retries - 1:
                f"Failed to get a response from the model after {config.retry_times} retries."
                time.sleep(retry_delay)
            else:
                print("Maximum retry attempts reached. Task failed.")
                return f"API call failed. Error: {e}"


def Judge_postive_Completion(
    messages,
    config: Config,
):
    """调用 GPT 模型进行正面判断"""
    return chatCompletion(messages, config)


def JudgeCompletion(
    messages,
    config: Config,
):
    """调用 OpenAI API 获取聊天补全结果"""
    model = config.judge_model
    temperature = config.temperature
    retry_times = config.retry_times
    round_sleep = config.round_sleep
    fail_sleep = config.fail_sleep
    api_key = config.gpt_api_key
    base_url = config.gpt_base_url

    # 初始化 OpenAI 客户端
    client = (
        OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else OpenAI(api_key=api_key)
    )

    response = None  # 初始化 response 变量
    for retry_time in range(retry_times):  # 尝试重试
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            break  # 成功后退出循环
        except Exception as e:
            print(f"Retry {retry_time + 1}/{retry_times} failed for {model}: {e}")
            time.sleep(fail_sleep)  # 重试前等待一段时间

    # 如果所有尝试均失败
    if response is None:
        raise RuntimeError(
            f"Failed to get a response from the model after {retry_times} retries."
        )

    # 成功后解析响应
    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)  # 等待指定的时间间隔

    return model_output
