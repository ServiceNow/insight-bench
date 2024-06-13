import requests

from types import SimpleNamespace

BASE_URL = {
    "mixtral": "https://26e90e35-003e-42b9-848e-71726531c929-8085.job.console.elementai.com/v1/completions",
    "llama-3-8b": "https://92b1b53d-191a-47b5-aef2-0a175c29857e-8085.job.console.elementai.com/v1/completions",
    "llama-3-70b": "https://b339c95b-27c9-4c22-97f8-c0b6de1113c7-8085.job.console.elementai.com/v1/completions",
    "dummy": "https://www.google.com",
}
AUTH_TOKEN = {
    "mixtral": "Bearer OMZQQbS-bwUMSGoeW-aVFg:M-By7BK9LShn4VSIJ8QOtfzreIlx4pBfgF9tIpzGs2E",
    "llama-3-8b": "Bearer ZFcFKFeeDiK2j65n08FAiQ:DE9GW-xetGCdjJWTkWPh9r71ooX7wB5WDGXxJHseYDY",
    "llama-3-70b": "Bearer p8EQ2DzLxroPRjTRhpFI7A:dC-bGld3gQ9Braog1VgghOjb39Wd5mCJsh7okwXPGWs",
    "dummy": "Bearer ZFcFKFeeDiK2j65n08FAiQ:DE9GW-xetGCdjJWTkWPh9r71ooX7wB5WDGXxJHseYD3423432",
}
MODELS = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "dummy": "dummy/dummy",
}


def infer_vllm(
    prompt,
    max_tokens=2000,
    end_point="llama-3-70b",
    top_logprobs=None,
    return_full_json=False,
    temperature=0,
):
    data = {
        "model": MODELS[end_point],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": top_logprobs,
    }
    headers = {
        "Content-Type": "application/json",
        "Cookie": "sessiona=1687876608.234.49.972136|78cabb3f310793e5a58a141fe9058709",
        "Authorization": f"{AUTH_TOKEN[end_point]}",
    }
    if end_point == "dummy":
        content = """
        Prompt 1 get dataset description
        <description>this is weird</description>

        Prompt 2 get question
        <question>What is this weird thing?</question>

        Prompt 3 generate code
        ```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Create dummy data
x_values = np.linspace(0, 10, 100).tolist()
y_values = np.sin(x_values).tolist()

x_data = {
    "type": "x_axis",
    "values": x_values
}

y_data = {
    "type": "y_axis",
    "values": y_values
}

stat_data = {
    "type": "stat",
    "mean": np.mean(y_values),
    "std_dev": np.std(y_values),
    "max": np.max(y_values),
    "min": np.min(y_values)
}

# Save dummy data to JSON files
with open('x_axis.json', 'w') as f:
    json.dump(x_data, f)

with open('y_axis.json', 'w') as f:
    json.dump(y_data, f)

with open('stat.json', 'w') as f:
    json.dump(stat_data, f)

# Create and save a dummy plot
plt.figure()
plt.plot(x_data['values'], y_data['values'])
plt.title('Dummy Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.savefig('plot.jpg')
plt.close()

print("Dummy data and plot have been saved.")
        ```

        Prompt 4 Get Insight
        <insight>
        <answer>
        This is a weird thing
        </answer>
        <justification>
        This is a weird thing
        </justification>
        </insight>

        Prompt 5 Get follow up questions
        <descriptive>
        <analysis>What is the output of this code?</analysis>
        <follow_up>What is the purpose of this code?</follow_up>
        </descriptive>

        <diagnostic>
        <analysis>What is the output of this code?</analysis>
        <follow_up>What is the purpose of this code?</follow_up>
        </diagnostic>
        
        <prescriptive>
        <analysis>What is the output of this code?</analysis>
        <follow_up>What is the purpose of this code?</follow_up>
        </prescriptive>

        <predictive>
        <analysis>What is the output of this code?</analysis>
        <follow_up>What is the purpose of this code?</follow_up>
        </predictive>

        Prompt 6 Select follow up question
        <question_id>0</question_id>

        """
        return content

    else:
        try:
            result = requests.post(
                BASE_URL[end_point], headers=headers, json=data
            ).json()
        except Exception as e:
            result = {
                "choices": [{"text": f"Error in generation here is the full json: {e}"}]
            }
    if return_full_json:
        return result
    try:
        return result["choices"][0]["text"]
    except KeyError as e:
        print(f"Error in making requests for {end_point}: {e}")
        print(result)
        return result["message"]
    except Exception as e:
        print(f"Error in making requests for {end_point}: {e}")
        print(result)
        return result["message"]


def main():
    prompt = "This is a sample prompt."
    result = infer_vllm(prompt, end_point="llama-3-70b")
    print(result)


if __name__ == "__main__":
    main()
