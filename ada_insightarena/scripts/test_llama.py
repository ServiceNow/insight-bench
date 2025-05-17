import requests

url = "https://d60c-38-104-226-50.ngrok-free.app/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "meta-llama/Meta-Llama-3.2-90B-Vision-Instruct",
    "prompt": "Explain the concept of diffusion models in machine learning.",
    "max_tokens": 200,
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
print(response.json())
