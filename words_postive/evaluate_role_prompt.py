from utils.detect_toxic_utils.sensitive_detect import (
    predict_sensitivity,
)  # 导入敏感性预测函数
from config import config_args # 导入配置信息
from role_different_prompt import user_context,llm_context # 导入不同的角色 prompt
from utils.data_util.data_utils import data_reader # 读取恶意输入


user_input = user_context
llm_input = llm_context
# 主程序
def main():
    # 1.读取数据文件中的每一行
    data = data_reader(config_args.data_path)
    assert len(data) == 520
    
    results = []
    
    for question in data:
        question = question.strip()  # 去除换行符
        
        # 替换占位符
        user_context = user_input.replace("{{## Question ##}}", question)
        llm_context = llm_input.replace("{## Question ##}", question)
        
        # 预测敏感性分数
        user_sensitive_score = predict_sensitivity(user_context)
        llm_sensitive_score = predict_sensitivity(llm_context)
        
        results.append({
            'question': question,
            'user_sensitive_score': user_sensitive_score,
            'llm_sensitive_score': llm_sensitive_score
        })
    
    # 打印结果
    for result in results:
        print(f"Question: {result['question']}")
        print(f"User Context Sensitive Score: {result['user_sensitive_score']:.4f}%")
        print(f"LLM Context Sensitive Score: {result['llm_sensitive_score']:.4f}%")
        print()  # 打印空行以分隔不同问题的结果

if __name__ == "__main__":
    main()



