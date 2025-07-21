import os
import pandas as pd
from insightbench.utils.agent_utils import analysis_nb_to_gt
from evaluation.metrics_utils import score_insight
from evaluation import metrics
import nbformat, re, json


def evaluate_insights(pred_insights, gt_insights, score_name="bleurt"):
    # compute score using score_method
    if score_name == "bleurt":
        score, score_dict = metrics.compute_bleurt_score(pred_insights, gt_insights)
    elif score_name == "g_eval":
        score, score_dict = metrics.compute_g_eval_o2m(pred_insights, gt_insights)
    elif score_name=="llama3_eval":
        score,score_dict = metrics.compute_llama3_eval_o2m(pred_insights,gt_insights)
    return score