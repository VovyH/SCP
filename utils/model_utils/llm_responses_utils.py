from utils.model_utils.llm_completion_utils import chatCompletion, claudeCompletion
from config import config_args

# 定义用于GPT模型的gpt_responses函数
def gpt_responses(Config, text: str):
    user_input = text  # 用户输入的文本
    user_message = {"role": "user", "content": user_input}  # 创建用户消息字典
    messages = []  # 初始化消息列表
    messages.append(user_message)  # 将用户消息添加到消息列表中
    # 调用chatCompletion函数获取GPT模型的响应
    model_output = chatCompletion(
        messages=messages,  # 消息列表
        config=config_args  # 配置对象
    )
    return model_output  # 返回GPT模型的响应

# 定义用于Claude模型的claude_responses函数
def claude_responses(config_args, text: str):
    user_input = text  # 用户输入的文本
    # 调用claudeCompletion函数获取Claude模型的响应
    model_output = claudeCompletion(
        user_input,                   # 输入提示符
        config=config_args  # 配置对象
    )
    return model_output  # 返回Claude模型的响应

# 定义用于Llama2模型的llama2_responses函数
def llama2_responses(config_args, text: str):
    pass  # Llama2模型的具体实现尚未完成

# 定义用于Mistral模型的mistral_responses函数
def mistral_responses(config_args, model, tokenizer, text: str):
    user_input = [
        {"role": "user", "content": text}  # 创建用户消息字典
    ]
    
    # 使用tokenizer将用户消息转换为模型所需的格式
    encodeds = tokenizer.apply_chat_template(user_input, return_tensors="pt")
    model_inputs = encodeds.to("cuda")  # 将输入数据移动到GPU
    model.to("cuda")  # 将模型移动到GPU
    
    # 生成新的tokens
    generated_ids = model.generate(
        model_inputs,                     # 模型输入
        pad_token_id=tokenizer.eos_token_id,  # 结束符号ID
        max_new_tokens=config_args.max_tokens,   # 最大生成token数量
        do_sample=True                    # 是否采样
    )
    decoded = tokenizer.batch_decode(generated_ids)  # 解码生成的tokens
    
    # 分割解码后的字符串以提取实际内容
    parts = decoded[0].split("[/INST] ")
    if len(parts) > 1:
        content_after_inst = parts[1]  # 提取[/INST]之后的内容
    else:
        content_after_inst = ""  # 如果没有找到[/INST]，则为空字符串
    model_output = content_after_inst.replace("</s>", "")  # 移除结束符号</s>
    
    return model_output  # 返回Mistral模型的响应