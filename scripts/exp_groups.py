EXP_GROUPS = {}

AGENT_LIST = ["gpt-4o", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125", "llama-3-70b"]

# DATASET_IDS = [14, 15, 18, 21]
DATASET_IDS = list(range(1, 32))
SLOPE_LIST = [1] + list(range(0, 100, 5))[1:]


EVAL_METRICS = [
    "rouge1",
    "g_eval",
    # "llama3_eval",
    # "g_eval_m2m",
    # # "bert_score",
    # "rouge_score",
    # "sbert_score",
    # "mixtral_eval",
]

# Quick Test
EXP_GROUPS["quick"] = []

for gen_engine in AGENT_LIST[:1]:
    for i in [1]:
        # for i in DATASET_IDS[:1]:
        EXP_GROUPS["quick"].append(
            {
                "gen_engine": gen_engine,
                "dataset_id": i,
                "use_goal": True,
                "do_sensitivity": False,
                "eval_metrics": EVAL_METRICS,
                "max_questions": 2,
                "branch_depth": 3,
            }
        )

# Baselines
EXP_GROUPS["baselines"] = []

for gen_engine in AGENT_LIST:
    for i in DATASET_IDS:
        for use_goal in [True, False]:
            for seed in range(3):
                EXP_GROUPS["baselines"].append(
                    {
                        "gen_engine": gen_engine,
                        "dataset_id": i,
                        "use_goal": use_goal,
                        "do_sensitivity": False,
                        "eval_metrics": EVAL_METRICS,
                        "max_questions": 3,
                        "branch_depth": 5,
                        "seed": seed,
                    }
                )

EXP_GROUPS["sensitivity"] = []

for gen_engine in AGENT_LIST:
    for s in SLOPE_LIST:
        for seed in range(3):
            EXP_GROUPS["sensitivity"].append(
                {
                    "gen_engine": gen_engine,
                    "dataset_id": 2,
                    "slope": s,
                    "use_goal": True,
                    "do_sensitivity": True,
                    "eval_metrics": EVAL_METRICS,
                    "max_questions": 3,
                    "branch_depth": 5,
                    "seed": seed,
                }
            )
