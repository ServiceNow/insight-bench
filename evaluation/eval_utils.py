from evaluation import prompts

import logging 
import numpy as np, pandas as time, re, os
import evaluate

import requests

import openai
from openai import OpenAI

from evaluate import load

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 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
def compute_g_eval(answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return compute_llm_eval(client, answer, gt_answer, model_name, top_logprobs)

def compute_llama3_eval(
    answer,
    gt_answer,
    model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    top_logprobs=None,
):
    """Compute LLaMA-3-Eval score between answer and gt_answer, with error handling."""
    # Initialize client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
    except Exception as e:
        logging.error("Failed to initialize OpenAI client: %s", e)
        raise RuntimeError(f"OpenAI client initialization error: {e}")

    # Run the evaluation
    try:
        return compute_llm_eval(
            client,
            answer,
            gt_answer,
            model_name,
            use_logprobs=True,
            top_logprobs=top_logprobs,
        )
    except Exception as e:
        logging.error("compute_llm_eval failed: %s", e)
        raise RuntimeError(f"LLaMA-3 eval computation error: {e}")

def compute_llm_eval(client, answer, gt_answer, model_name="gpt-4o", detailed: bool = True,top_logprobs=None,max_retries: int = 5):
    template, system_message = prompts.get_g_eval_prompt(method="basic")

    prompt = template.format(answer=answer, gt_answer=gt_answer)
    if model_name.startswith("nvidia/llama-3.1-nemotron-ultra-253b"):
        system_message += f"\n\nDetailed thinking = {'ON' if detailed else 'OFF'}."
    retries=0
    backoff=10
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
        except openai.error.RateLimitError as e:
            retries += 1
            if retries > max_retries:
                logging.error("Rate limit exceeded after %d retries", retries)
                raise
            logging.warning("RateLimitError, retry %d/%d after %ds...", retries, max_retries, backoff)
            time.sleep(backoff)
            backoff *= 2

        except openai.error.APIError as e:
            retries += 1
            if retries > max_retries:
                logging.error("API error after %d retries: %s", retries, e)
                raise
            logging.warning("APIError (%s), retry %d/%d after %ds...", e, retries, max_retries, backoff)
            time.sleep(backoff)
            backoff *= 2

        except Exception as e:
            # non-OpenAI errors: probably a bug—don't retry endlessly
            logging.error("Unexpected error in compute_llm_eval: %s", e, exc_info=True)
            raise



def compute_bleurt_score(answer, gt_answer, **kwargs):
    """
    Compute BLEURT score between answer and gt_answer.
    """
    bleurt = load("bleurt", config_name="bleurt-large-512")

    result = bleurt.compute(
        predictions=[answer],
        references=[gt_answer]
    )
    
    return result["scores"][0]
