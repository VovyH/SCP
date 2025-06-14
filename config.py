from dataclasses import dataclass, asdict
import json

@dataclass
class Config:

    # GPT模型相关参数
    gpt_model: str = "gpt-4-0613"
    gpt_api_key: str = "xxx"
    gpt_base_url: str = "https://openai.api2d.net/v1"

    # Claude模型相关参数
    claude_model: str = "claude-3-5-sonnet"
    claude_api_key: str = "xxxxF"
    claude_base_url: str = "https://oa.api2d.net/claude"

    # Llama2模型相关参数
    llama2_model: str = "llama2-model"
    llama2_api_key: str = ""
    llama2_base_url: str = None

    # Mistral模型相关参数
    mistral_model: str = "mistral-model"
    mistral_tokenizer_name: str = "mistral-tokenizer"

    # 公共参数
    temperature: float = 0.7
    max_tokens: int = 256
    retry_times: int = 10
    round_sleep: float = 1.0
    fail_sleep: float = 2.0

    # 数据集和设备相关参数
    devices: str = "0"
    # 数据集路径
    data_path: str = r"xxxx"
    # 模式
    mode = "test"
    # 测试模型
    test_model: str = "claude-3-5-sonnet-20240620"
    judge_model: str = "gpt-4-0613"
    refine_model: str = test_model
    
    # 参数
    top_k = 50
    max_tokens = 1024
    repetition_penalty = 1.0
    temperature = 0.0

# 示例配置实例
config_args = Config()
# 使用 asdict 将 dataclass 转换为字典
config_dict = asdict(config_args)

# 序列化为 JSON
config_json = json.dumps(config_dict, indent=4)
print(config_args)
