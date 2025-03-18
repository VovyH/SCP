import requests
import json

url = 'https://api.mixrai.com/v1/chat/completions'
# 'xxx' is your API key, 换成你的令牌
api_key = 'sk-Entan08ZP4morw94Ac11504946F4453cBa58Cd7360Ae5eE2'

payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}]
})

headers = {
    "Accept": "application/json",
    "Authorization": api_key,
    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    "Content-Type": "application/json",
}

response = requests.request('POST', url, headers=headers, data=payload)

print(response.text)
print('----下面是API Key----')
print(api_key)