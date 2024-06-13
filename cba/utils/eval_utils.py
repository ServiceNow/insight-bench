from cba import prompts

import numpy as np, pandas as pd, time, re, os
import evaluate

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
from cba.vllm import infer_vllm

# from openai import OpenAI
import openai
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from functools import partial
from openai import OpenAI
from cba.utils.agent_utils import convert_messages_to_text

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# BERT_SCORE = evaluate.load("bertscore")
ROUGE_SCORE = evaluate.load("rouge")


def compute_g_eval(answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
    client = OpenAI(api_key=OPENAI_API_KEY)
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


def compute_g_eval_m2m(
    pred_insights, gt_insights, model_name="gpt-4o", top_logprobs=None
):
    """Does many-to-many matching of provided and gt insights"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    template = prompts.G_EVAL_M2M_TEMPLATE
    pred_insights_formatted = "\n".join(
        [f"{idx+1}. {a}" for idx, a in enumerate(pred_insights)]
    )
    gt_answers_formatted = "\n".join(
        [f"{idx+1}. {a}" for idx, a in enumerate(gt_insights)]
    )
    prompt = template.format(
        pred_list=pred_insights_formatted, gt_list=gt_answers_formatted
    )
    fail_count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.G_EVAL_M2M_SYSTEM_MESSAGE,
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
            matched_responses = []
            for line in response.choices[0].message.content.splitlines():
                if line.strip().isdigit():
                    matched_responses.append(int(line.strip()))
                else:  # try to capture 1. -1 type outputs
                    matched_responses.append(
                        int(re.sub(r"\d\.\s(.+)", r"\1", line).strip())
                    )
            scores_dict = []
            for id, mid in enumerate(matched_responses):
                mid = mid - 1 if mid > 0 else np.random.choice(len(pred_insights))
                score = (
                    compute_g_eval(
                        pred_insights[mid],
                        gt_insights[id],
                        model_name,
                        top_logprobs,
                    )
                    / 10.0
                )
                scores_dict.append(
                    {
                        "pred_insight": pred_insights[mid],
                        "gt_insight": gt_insights[id],
                        "score": score,
                    }
                )
            score = np.mean([score["score"] for score in scores_dict])
            return score, scores_dict
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"Error occured: {e}, Retrying")
            if fail_count <= 5:
                fail_count += 1
                continue
            print("Retries exhausted, returning random match G-Eval results")
            # return random matching results
            scores_dict = []
            for id in range(len(gt_insights)):
                mid = np.random.choice(len(pred_insights))
                score = (
                    compute_g_eval(
                        pred_insights[mid],
                        gt_insights[id],
                        top_logprobs,
                    )
                    / 10.0
                )
                scores_dict.append(
                    {
                        "pred_insight": pred_insights[mid],
                        "gt_insight": gt_insights[id],
                        "score": score,
                    }
                )
            score = np.mean([score["score"] for score in scores_dict])
            return score, scores_dict


def compute_llama3_eval_m2m(
    pred_insights, gt_insights, model_name="llama-3-70b", top_logprobs=None
):
    """Does many-to-many matching of provided and gt insights"""
    chat = partial(
        infer_vllm,
        max_tokens=50,
        end_point=model_name,
        top_logprobs=None,
        return_full_json=False,
    )

    system_message_prompt = prompts.G_EVAL_M2M_SYSTEM_MESSAGE
    human_message_prompt = prompts.G_EVAL_M2M_TEMPLATE
    pred_insights_formatted = "\n".join(
        [f"{idx+1}. {a}" for idx, a in enumerate(pred_insights)]
    )
    gt_answers_formatted = "\n".join(
        [f"{idx+1}. {a}" for idx, a in enumerate(gt_insights)]
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    formatted_messages_str = chat_prompt.format_prompt(
        pred_list=pred_insights_formatted,
        gt_list=gt_answers_formatted,
    ).to_messages()
    fail_count = 0
    while True:
        try:
            formatted_messages = convert_messages_to_text(formatted_messages_str)
            response = chat(formatted_messages)
            matched_responses = []
            for line in response.splitlines():
                if line.strip().isdigit():
                    matched_responses.append(int(line.strip()))
                else:  # try to capture 1. -1 type outputs
                    matched_responses.append(
                        int(re.sub(r"\d\.\s(.+)", r"\1", line).strip())
                    )
            scores_dict = []
            for id, mid in enumerate(matched_responses):
                mid = mid - 1 if mid > 0 else np.random.choice(len(pred_insights))
                score = (
                    compute_llama3_eval(
                        pred_insights[mid],
                        gt_insights[id],
                        top_logprobs,
                    )
                    / 10.0
                )
                scores_dict.append(
                    {
                        "pred_insight": pred_insights[mid],
                        "gt_insight": gt_insights[id],
                        "score": score,
                    }
                )
            score = np.mean([score["score"] for score in scores_dict])
            return score, scores_dict
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"Error occured: {e}, Retrying")
            if fail_count <= 5:
                fail_count += 1
                continue
            print("Retries exhausted, returning random match LLama Eval results")
            # return random matching results
            scores_dict = []
            for id in range(len(gt_insights)):
                mid = np.random.choice(len(pred_insights))
                score = (
                    compute_llama3_eval(
                        pred_insights[mid],
                        gt_insights[id],
                        top_logprobs,
                    )
                    / 10.0
                )
                scores_dict.append(
                    {
                        "pred_insight": pred_insights[mid],
                        "gt_insight": gt_insights[id],
                        "score": score,
                    }
                )
            score = np.mean([score["score"] for score in scores_dict])
            return score, scores_dict


def compute_rouge_score(answer, gt_answer, **kwargs):
    """Compute ROUGE-1 between answer and gt_answer"""
    return ROUGE_SCORE.compute(
        predictions=[answer],
        references=[gt_answer],
        rouge_types=["rouge1"],
    )["rouge1"]


def compute_llama3_eval(answer, gt_answer, top_logprobs=None, **kwargs):
    return compute_mixtral_eval(
        answer, gt_answer, top_logprobs, model_name="llama-3-70b", **kwargs
    )


def compute_mixtral_eval(
    answer,
    gt_answer,
    top_logprobs=None,
    model_name="mixtral",
    **kwargs,
):
    chat = partial(
        infer_vllm,
        max_tokens=300,
        end_point=model_name,
        top_logprobs=top_logprobs,
        return_full_json=True,
    )
    system_template, prompt_template = prompts.get_g_eval_prompt(method="basic")
    # Define system and human message prompts
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)

    # Combine templates into a chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Generate the chat completion from the formatted messages
    formatted_messages_str = chat_prompt.format_prompt(
        answer=answer, gt_answer=gt_answer
    ).to_messages()

    try:
        formatted_messages = convert_messages_to_text(formatted_messages_str)
        response = chat(formatted_messages)
        if not top_logprobs:
            score = response
        else:
            # get the index in response where we have the rating
            rating_str = re.findall(
                r"<rating>(\d+)</rating>", response["choices"][0]["text"]
            )[0]
            rating_idx_in_response = response["choices"][0]["logprobs"]["tokens"].index(
                rating_str
            )
            response = response["choices"][0]["logprobs"]["top_logprobs"][
                rating_idx_in_response
            ]

            # convert logprobs to probs
            probs = [np.exp(logprob) for token, logprob in response.items()]
            # renormalize probs to sum to 1
            probs = [obj / sum(probs) for obj in probs]
            ratings = [
                float(token) if token.isdigit() else 0 for token, _ in response.items()
            ]
            # final score
            score = sum([a * b for a, b in zip(ratings, probs)])
            score = float(score)
        return score
    except Exception as e:
        print(f"Error in {model_name} eval: {e}")
        return 0
