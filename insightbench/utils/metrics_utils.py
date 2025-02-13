from insightbench.utils.eval_utils import (
    compute_rouge_score,
    compute_g_eval,
    compute_llama3_eval,
)


def score_insight(gt_insight, pred_insight, score_name):
    """
    Get the scoring function based on the score_name.

    Returns:
    --------
    score (float): The score of the prediction based on the ground truth.
    """
    if score_name == "rouge1":
        score = compute_rouge_score(pred_insight, gt_insight)
    elif score_name == "g_eval":
        score = compute_g_eval(pred_insight, gt_insight, top_logprobs=5) / 10.0
    elif score_name == "llama3_eval":
        score = compute_llama3_eval(pred_insight, gt_insight, top_logprobs=5) / 10.0
    else:
        raise ValueError(f"Unknown score_name: {score_name}")

    # score has to be between 0 and 1
    assert score >= 0 and score <= 1

    return score
