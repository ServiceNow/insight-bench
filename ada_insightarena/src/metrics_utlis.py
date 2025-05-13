import openai
from openai import OpenAI
import numpy as np, pandas as time, re, os
import evaluate
from src.prompts import get_g_eval_prompt

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def compute_rouge_score(answer, gt_answer, **kwargs):
    """Compute ROUGE-1 between answer and gt_answer"""
    ROUGE_SCORE = evaluate.load("rouge")

    return ROUGE_SCORE.compute(
        predictions=[answer],
        references=[gt_answer],
        rouge_types=["rouge1"],
    )["rouge1"]

def compute_g_eval(answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
    client = OpenAI(api_key=OPENAI_API_KEY)
    template, system_message = get_g_eval_prompt(method="basic")

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