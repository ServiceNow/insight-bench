from collections import defaultdict
import numpy as np
from insightbench.utils import metrics_utils as mu
from insightbench.utils import eval_utils as eu
from tqdm import tqdm


def compute_rouge(pred_insights, gt_insights, return_scores=False):
    """
    Compute the ROUGE score for a list of predictions and ground truths.

    Args:
    -----
    pred_insights (List[str]): The list of predicted insights.
    gt_insights (List[str]): The list of ground truth insights.

    Returns:
    --------
    score (float): The ROUGE score.
    """
    # Compute the ROUGE score for each prediction and ground truth pair
    score_dict = defaultdict(list)
    for gt_id, gt_insight in enumerate(gt_insights):
        for pred_id, pred_insight in enumerate(pred_insights):
            score = mu.score_insight(gt_insight, pred_insight, score_name="rouge1")
            score_dict[gt_id].append(score)

    best_pred_ids = [np.argmax(scores) for scores in score_dict.values()]
    score_dict = [
        {
            "pred_insight": pred_insights[best_pred_ids[gt_id]],
            "gt_insight": gt_insights[gt_id],
            "score": scores[best_pred_ids[gt_id]],
        }
        for gt_id, scores in score_dict.items()
    ]
    score = np.mean([score["score"] for score in score_dict])
    return score, score_dict


def compute_g_eval_m2m(pred_insights, gt_insights, return_scores=False):
    """
    Compute the G-Eval score for a list of predictions and ground truths.

    Args:
    -----
    pred_insights (List[str]): The list of predicted insights.
    gt_insights (List[str]): The list of ground truth insights.

    Returns:
    --------
    score (float): The G-Eval score.
    """
    # Compute the G-Eval (many-to-many version) score for each prediction and ground truth pair
    return eu.compute_g_eval_m2m(pred_insights, gt_insights, top_logprobs=5)


def compute_g_eval_o2m(pred_insights, gt_insights, return_scores=False):
    """
    Compute the G-Eval score for a list of predictions and ground truths.

    Args:
    -----
    pred_insights (List[str]): The list of predicted insights.
    gt_insights (List[str]): The list of ground truth insights.

    Returns:
    --------
    score (float): The G-Eval score.
    """
    # Compute the G-Eval (many-to-many version) score for each prediction and ground truth pair
    # Compute the G-Eval (one-to-many version) score for each prediction and ground truth pair
    scores_list = defaultdict(list)

    pbar = tqdm(enumerate(gt_insights), leave=False, total=len(gt_insights))
    for idx, insight in pbar:
        for pred_insight in pred_insights:
            scores_list[idx].append(
                eu.compute_g_eval(pred_insight, insight, top_logprobs=5) / 10.0
            )
    score_dict = []
    for gt_id in scores_list:
        best_pred_id = np.argmax(scores_list[gt_id])
        score_dict.append(
            {
                "pred_insight": pred_insights[best_pred_id],
                "gt_insight": gt_insights[gt_id],
                "score": scores_list[gt_id][best_pred_id],
            }
        )
    score = np.mean([score["score"] for score in score_dict])
    return score, score_dict


def compute_llama3_eval_m2m(pred_insights, gt_insights, return_scores=False):
    """
    Compute the G-Eval score for a list of predictions and ground truths.

    Args:
    -----
    pred_insights (List[str]): The list of predicted insights.
    gt_insights (List[str]): The list of ground truth insights.

    Returns:
    --------
    score (float): The G-Eval score.
    """
    return eu.compute_llama3_eval_m2m(pred_insights, gt_insights, top_logprobs=5)


def compute_llama3_eval_o2m(pred_insights, gt_insights, return_scores=False):
    """
    Compute the LLaMA-3-Eval score for a list of predictions and ground truths.

    Args:
    -----
    pred_insights (List[str]): The list of predicted insights.
    gt_insights (List[str]): The list of ground truth insights.

    Returns:
    --------
    score (float): The LLaMa-3-Eval score.
    """
    # Compute the LLaMa-3-Eval (one-to-many version) score for each prediction and ground truth pair
    scores_list = defaultdict(list)
    pbar = tqdm(enumerate(gt_insights), leave=False, total=len(gt_insights))
    for idx, insight in pbar:
        for pred_insight in pred_insights:
            scores_list[idx].append(
                eu.compute_llama3_eval(pred_insight, insight, top_logprobs=5) / 10.0
            )
    scores_list = dict(scores_list)
    score_dict = []
    for gt_id in scores_list:
        best_pred_id = np.argmax(scores_list[gt_id])
        score_dict.append(
            {
                "pred_insight": pred_insights[best_pred_id],
                "gt_insight": gt_insights[gt_id],
                "score": scores_list[gt_id][best_pred_id],
            }
        )
    score = np.mean([score["score"] for score in score_dict])
    return score, score_dict
