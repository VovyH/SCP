import os
from http import HTTPStatus
import dashscope

# 设置 DashScope API Key
dashscope.api_key = "sk-d07d9d5c4d8d4158abbaf45a40c10042"  # 替换为你的实际 API Key

def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '介绍下故宫？'}]
    response = dashscope.Generation.call(
        model='llama3-8b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    call_with_messages()