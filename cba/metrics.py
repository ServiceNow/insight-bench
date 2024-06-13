from collections import defaultdict
import numpy as np
from cba.utils import metrics_utils as mu
from cba.utils import eval_utils as eu
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


if __name__ == "__main__":
    gt_insights = gt_boxes = [
        "Prioritize tasks, delegate wisely.",
        "Clear goals boost productivity.",
        "Effective communication is key.",
        "Foster a culture of trust.",
        "Recognize and reward achievements.",
    ]

    pred_insights = pred_boxes = [
        "Delegate tasks effectively.",
        "Set clear, achievable goals.",
        "Communication prevents misunderstandings.",
        "Build a trusting environment.",
        "Celebrate and acknowledge success.",
    ]

    score = compute_mAP(gt_boxes, pred_boxes, score_method="rouge1")
    print("mAP: {:.4f}".format(score))

# if __name__ == "__main__":

#     with open("/mnt/home/projects/research-cba/data/gt_boxes.json") as infile:
#         gt_boxes = json.load(infile)

#     with open("/mnt/home/projects/research-cba/data/pred_boxes.json") as infile:
#         pred_boxes = json.load(infile)

#     use_cba = True
#     if use_cba:
#         # gt_boxes = gt_boxes["img_00172.png"]
#         # pred_boxes = pred_boxes["img_00172.png"]["boxes"]
#         gt_insights = gt_boxes = [
#             "Prioritize tasks, delegate wisely.",
#             "Clear goals boost productivity.",
#             "Effective communication is key.",
#             "Foster a culture of trust.",
#             "Recognize and reward achievements.",
#         ]

#         pred_insights = pred_boxes = [
#             "Delegate tasks effectively.",
#             "Set clear, achievable goals.",
#             "Communication prevents misunderstandings.",
#             "Build a trusting environment.",
#             "Celebrate and acknowledge success.",
#         ]

#     # Runs it for one IoU threshold
#     iou_thr = 0.7
#     start_time = time.time()
#     data = get_cba_avg_precision_at_iou(
#         gt_boxes, pred_boxes, iou_thr=iou_thr, use_cba=use_cba
#     )
#     end_time = time.time()
#     print("Single IoU calculation took {:.4f} secs".format(end_time - start_time))
#     print("avg precision: {:.4f}".format(data["avg_prec"]))

#     start_time = time.time()
#     ax = None
#     avg_precs = []
#     iou_thrs = []
#     for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
#         data = get_cba_avg_precision_at_iou(
#             gt_boxes, pred_boxes, iou_thr=iou_thr, use_cba=use_cba
#         )
#         avg_precs.append(data["avg_prec"])
#         iou_thrs.append(iou_thr)

#         precisions = data["precisions"]
#         recalls = data["recalls"]
#         ax = plot_pr_curve(
#             precisions,
#             recalls,
#             label="{:.2f}".format(iou_thr),
#             color=COLORS[idx * 2],
#             ax=ax,
#         )

#     # prettify for printing:
#     avg_precs = [float("{:.4f}".format(ap)) for ap in avg_precs]
#     iou_thrs = [float("{:.4f}".format(thr)) for thr in iou_thrs]
#     print("map: {:.2f}".format(100 * np.mean(avg_precs)))
#     print("avg precs: ", avg_precs)
#     print("iou_thrs:  ", iou_thrs)
#     plt.legend(loc="upper right", title="IOU Thr", frameon=True)
#     for xval in np.linspace(0.0, 1.0, 11):
#         plt.vlines(xval, 0.0, 1.1, color="gray", alpha=0.3, linestyles="dashed")
#     end_time = time.time()
#     print(
#         "\nPlotting and calculating mAP takes {:.4f} secs".format(end_time - start_time)
#     )
#     plt.show()
