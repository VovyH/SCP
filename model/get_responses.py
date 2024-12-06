import os
from utils.data_util.data_utils import data_reader
from utils.model_utils.llm_responses_utils import (
    gpt_responses,
    claude_responses,
    mistral_responses,
    llama2_responses,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config_args


def main(config_args):

    # 1.读取需要越狱的数据
    data = data_reader(config_args.data_path)
    
    # 2.进入方法的标志
    chemotherapy_method = False
    
    # 断言数据长度为520（可选）
    # assert len(data) == 520

    # 2.根据测试模式或评估模式决定处理哪些数据
    if config_args.mode == "test":
        data_to_process = data[:1]  # 仅处理第一条数据
    else:
        data_to_process = data  # 处理所有数据

    # 3.选择模型model和分词器tokenizer
    try:
        if "mistral" in config_args.mistral_model:
            model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/Mistral-7B-Instruct-v0.2"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif "llama2" in config_args.llama2_model:
            model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/Mistral-7B-Instruct-v0.2"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model:{e},so,Choosing not Opening model.")
        model = None
        tokenizer = None

    # TODO: 初始化句子：利用wordnet修改data_to_process，具体首先包装成疑问句，然后将动词置反
    

    # 4.根据不同的模型进行评测
    for idx, item in enumerate(data_to_process):
        # while(chemotherapy_method):
        #     # TODO:进行chemotherapy_method，以生成恶意句子
            
        harmful_data = item
        if "gpt" in config_args.test_model:  # GPT生成
            model_output = gpt_responses(config_args, harmful_data)
        elif "claude" in config_args.test_model:  # Claude
            model_output = claude_responses(config_args, harmful_data)
        elif "mistral" in config_args.test_model:  # mistral
            model_output = mistral_responses(
                config_args, model, tokenizer, harmful_data
            )
        elif "llama2" in config_args.test_model:  # llama2
            model_output = llama2_responses(
                config_args, model, tokenizer, harmful_data
            )
        else:
            raise ValueError(f"Unsupported test model: {config_args.test_model}")
        # TODO：用关键词列表评价model_output，若包含关键词列表中的元素则将chemotherapy_method置为true，
        
        
        # 5.打印当前处理数据信息
        print(
            "\n################################\n\n"
            f"***** Current Data: {idx+1}/{len(data_to_process)} *****\n\n"
            f"***** Harmful Data *****\n{harmful_data}\n\n"
            f"***** Test Model *****\n{config_args.test_model}\n\n"
            f"***** Response of {config_args.test_model} *****\n\n"
            f"{model_output}\n\n"
            "################################\n"
        )
        # 6.日志 item 记录到结果集合中
        item_new = {}
        item_new['idx']  = idx
        item_new['harmful_behavior']  = harmful_data
        item_new['test_model'] = config_args.test_model
        item_new['model_output'] = model_output
        
        
if __name__ == "__main__":
    main(config_args)