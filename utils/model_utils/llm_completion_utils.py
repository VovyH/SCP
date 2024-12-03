from openai import OpenAI  # 导入OpenAI客户端库
import time  # 导入time库以便使用sleep功能
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT  # 导入Anthropic客户端库及其常量
from config import Config

# 定义用于GPT模型的chatCompletion函数
def chatCompletion(
    messages,
    config: Config,  # 配置对象
):
    model = config.gpt_model
    temperature = config.temperature
    retry_times = config.retry_times
    round_sleep = config.round_sleep
    fail_sleep = config.fail_sleep
    api_key = config.gpt_api_key
    base_url = config.gpt_base_url
    # 根据是否提供了base_url来初始化OpenAI客户端
    if base_url is None:
        client = OpenAI(api_key=api_key)  # 初始化OpenAI客户端，不指定基础URL
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)  # 初始化OpenAI客户端，并指定基础URL
    
    try:
        # 调用OpenAI API获取聊天补全结果
        response = client.chat.completions.create(
                model=model,  # 指定使用的模型
                messages=messages,  # 提供消息列表
                temperature=temperature  # 设置生成内容的随机性
        )
    except Exception as e:
        print(e)  # 打印异常信息
        for retry_time in range(retry_times):  # 循环尝试次数
            retry_time = retry_time + 1  # 计数器加一
            print(f"{model} Retry {retry_time}")  # 打印重试信息
            time.sleep(fail_sleep)  # 等待一段时间后重试
            try:
                # 再次调用OpenAI API获取聊天补全结果
                response = client.chat.completions.create(
                    model=model,  # 指定使用的模型
                    messages=messages,  # 提供消息列表
                    temperature=temperature  # 设置生成内容的随机性
                )
                break  # 成功后跳出循环
            except:
                continue  # 继续下一次循环
    
    # 获取并清理API返回的消息内容
    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)  # 等待指定的时间间隔
    
    return model_output  # 返回处理后的消息内容

# 定义用于Claude模型的claudeCompletion函数
def claudeCompletion(
    prompt,         # 输入提示符
    config: Config,  # 配置对象
):
    model = config.gpt_model
    temperature = config.temperature
    retry_times = config.retry_times
    round_sleep = config.round_sleep
    fail_sleep = config.fail_sleep
    api_key = config.gpt_api_key
    base_url = config.claude_base_url
    max_tokens = config.max_tokens
    
    # 根据是否提供了base_url来初始化Anthropic客户端
    if base_url is None:
        client = Anthropic(api_key=api_key)  # 初始化Anthropic客户端，不指定基础URL
    else:
        client = Anthropic(base_url=base_url, auth_token=api_key)  # 初始化Anthropic客户端，并指定基础URL和认证令牌
    
    try:
        # 调用Anthropic API获取补全结果
        completion = client.completions.create(
            model=model,  # 指定使用的模型
            max_tokens_to_sample=max_tokens,  # 设置最大生成token数量
            temperature=temperature,  # 设置生成内容的随机性
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"  # 构建提示符，包含人类和AI的标识
            )
    except Exception as e:
        print(e)  # 打印异常信息
        for retry_time in range(retry_times):  # 循环尝试次数
            retry_time = retry_time + 1  # 计数器加一
            print(f"{model} Retry {retry_time}")  # 打印重试信息
            time.sleep(fail_sleep)  # 等待一段时间后重试
            try:
                # 再次调用Anthropic API获取补全结果
                completion = client.completions.create(
                model=model,  # 指定使用的模型
                max_tokens_to_sample=max_tokens,  # 设置最大生成token数量
                temperature=temperature,  # 设置生成内容的随机性
                prompt=prompt  # 提供提示符
                )
                break  # 成功后跳出循环
            except:
                continue  # 继续下一次循环
    
    # 获取并清理API返回的补全内容
    model_output = completion.completion.strip()
    time.sleep(round_sleep)  # 等待指定的时间间隔
    
    return model_output  # 返回处理后的补全内容



