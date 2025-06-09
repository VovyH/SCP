from urllib import response
from openai import OpenAI
import requests
import pandas as pd
from tqdm import tqdm
import requests
import http.client
import concurrent.futures
import json

def Aime_POSD(message,model_type):
    if model_type == "gpt4":

        url = "https://xiaoai.plus/v1/chat/completions"
        payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False,
        "model": "deepseek-r1",
        "temperature": 0.0,
        "max_tokens": 5096
        })
        headers = {
            "Content-Type": "application/json",
            'Authorization': 'sk-ZkY44LjiNh2aEiY7WY0wx8OnOQLqDfTyNxF4hLr0bcaqBIbX',
        }
        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} with message: {response.text}")
                return ""
            
            response_json = response.json()
            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(response_text)
            return response_text
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text}")
            return ""
        return response_text
    else:
        client = OpenAI(
            base_url='https://xiaoai.plus/v1',
            # sk-xxx替换为自己的key
            api_key='sk-ZkY44LjiNh2aEiY7WY0wx8OnOQLqDfTyNxF4hLr0bcaqBIbX'
        )

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
        )
        response = completion.choices[0].message
        print(response)
    
def Process_Aime(aime_csv_path, result_csv_path, model_type):
    # 读取AIME数据集
    df = pd.read_csv(aime_csv_path)
    correct_count = 0
    results = []  # 用于存储每次评估的结果
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4-0613 AIME Origin"):
        problem = row['input']
        answer = row['output']
        model_response = Aime_POSD(problem,model_type)
        is_correct = evaluate_response(problem, model_response, answer, model_type)
        if is_correct:
            correct_count += 1
        # 保存当前结果
        results.append({
            'input': problem,
            'model_response': model_response,
            "answer": answer,
            'is_correct': is_correct
        })
    pd.DataFrame(results).to_csv(result_csv_path, index=False)
    print(f"correct_count = {correct_count}")

def evaluate_response(problem, model_response, correct_answer, model_type):
    """
    评估模型回答是否正确
    """
    system_prompt = f"""
    You are an intelligent AI assistant. Please evaluate the model's response to the following question is correct:
    
    Question: {problem}
    Model's Response: {model_response}
    Correct Answer: {correct_answer}
    
    If the model's response is consistent with or semantically equivalent to the correct answer, return 'True', otherwise return 'False'.
    """
    
    # 使用process_single_text方法处理评估
    evaluation_result = Aime_POSD(system_prompt,model_type)
    return "true" in evaluation_result.__str__().lower()

if __name__ == "__main__":
    aime_csv_path = r"e:\我的论文和代码\Chemotherapy\data\aime_2024_problems_processed.csv"
    result_csv_path = r"e:\我的论文和代码\Chemotherapy\defense\aime_deepseek_evaluation.csv"
    
    Process_Aime(aime_csv_path, result_csv_path, model_type="gpt4")