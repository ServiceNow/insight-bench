from evaluation import prompts

import numpy as np, pandas as time, re, os
import evaluate

import requests

import openai
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def compute_g_eval(answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return compute_llm_eval(client, answer, gt_answer, model_name, top_logprobs)


def is_llama_running(model_name):
    status_code = requests.post(
        "http://0.0.0.0:8085/v1/completions",
        json={"prompt": "Hello!", "model": model_name},
        headers={
            "Content-Type": "application/json",
            "Cookie": "sessiona=1687876608.234.49.972136|78cabb3f310793e5a58a141fe9058709",
            "Authorization": "EMPTY",
        },
    ).status_code
    return status_code == 200


def compute_llama3_eval(
    answer, gt_answer, model_name="meta-llama/Meta-Llama-3-70B", top_logprobs=None
):
    """Compute LLaMA-3-Eval score between answer and gt_answer"""
    # check if llama3 is running locally
    if is_llama_running(model_name):
        client = OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8085/v1/")
        return compute_llm_eval(client, answer, gt_answer, model_name, top_logprobs)
    else:
        raise RuntimeError(
            """
To use LLaMA-3-Eval, please first host a LLaMA-3 model locally using the vllm library:
```
pip install vllm
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model meta-llama/Meta-Llama-3-70B --tensor-parallel-size 8 --load-format safetensors --port 8085 --dtype half --gpu-memory-utilization 0.8 --max-model-len 8000 --enforce-eager
```
"""
        )


def compute_llm_eval(client, answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
    template, system_message = prompts.get_g_eval_prompt(method="basic")

    prompt = template.format(answer=answer, gt_answer=gt_answer)
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=50,
                top_p=1,
                logprobs=bool(top_logprobs),
                top_logprobs=top_logprobs,
            )
            if not top_logprobs:
                score = response.choices[0].message.content
            else:
                # get the index in response where we have the rating
                rating_str = re.findall(
                    r"<rating>(\d+)</rating>", response.choices[0].message.content
                )[0]
                tokens = [o.token for o in response.choices[0].logprobs.content]
                rating_idx_in_response = tokens.index(rating_str)
                response = (
                    response.choices[0]
                    .logprobs.content[rating_idx_in_response]
                    .top_logprobs
                )
                # convert logprobs to probs
                probs = [np.exp(obj.logprob) for obj in response]
                # renormalize probs to sum to 1
                probs = [obj / sum(probs) for obj in probs]
                ratings = [
                    float(obj.token) if obj.token.isdigit() else 0 for obj in response
                ]
                # final score
                score = sum([a * b for a, b in zip(ratings, probs)])
            try:
                score = float(score)
            except ValueError:
                score = float(score.splitlines()[0])
            except:
                score = 0
            return score
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")


def compute_rouge_score(answer, gt_answer, **kwargs):
    """Compute ROUGE-1 between answer and gt_answer"""
    ROUGE_SCORE = evaluate.load("rouge")

    return ROUGE_SCORE.compute(
        predictions=[answer],
        references=[gt_answer],
        rouge_types=["rouge1"],
    )["rouge1"]
