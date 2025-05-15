EXP_GROUPS = {}

EXP_GROUPS["insights_insightarena"] = []
EXP_GROUPS["insights_insightarena"] += [
    {
        "challenge": "mid",
        "model": "gpt-4o",
        "eval_mode": "insights",
        "with_skills": 1,
        "data_path": "data"
    }
]

# define the insights experiments
EXP_GROUPS["insights_wo_skills"] = []
for challenge in ["mid"]:
    EXP_GROUPS["insights_wo_skills"] += [
        {
            "challenge": "mid",
            "model": "gpt-4o",
            "eval_mode": "insights",
            "with_skills": 0,
        }
    ]

EXP_GROUPS["insights_wo_skills_sai_latest"] = []
for challenge in ["toy"]:
    EXP_GROUPS["insights_wo_skills_sai_latest"] += [
        {
            "challenge": "toy",
            "model": "gpt-4o",
            "eval_mode": "insights",
            "with_skills": 0,
        }
    ]


EXP_GROUPS["insights_w_skills_sai_latest"] = []
for challenge in ["toy"]:
    EXP_GROUPS["insights_w_skills_sai_latest"] += [
        {
            "challenge": "toy",
            "model": "gpt-4o",
            "eval_mode": "insights",
            "with_skills": 1,
        }
    ]

EXP_GROUPS["insights_w_skills"] = []
for challenge in ["mid"]:
    EXP_GROUPS["insights_w_skills"] += [
        {
            "challenge": "mid",
            "model": "gpt-4o",
            "eval_mode": "insights",
            "with_skills": 1,
        }
    ]

EXP_GROUPS["insights_w_skills_only"] = []
for challenge in ["toy"]:
    EXP_GROUPS["insights_w_skills_only"] += [
        {
            "challenge": "toy",
            "model": "gpt-4o",
            "eval_mode": "insights_only",
            "with_skills": 1,
        }
    ]

EXP_GROUPS["insights_w_skills_only_planning"] = []
for challenge in ["toy"]:
    EXP_GROUPS["insights_w_skills_only_planning"] += [
        {
            "challenge": "toy",
            "model": "gpt-4o",
            "eval_mode": "insights_only",
            "with_skills": 1,
        }
    ]

EXP_GROUPS["insights_wo_skills_only"] = []
for challenge in ["toy"]:
    EXP_GROUPS["insights_wo_skills_only"] += [
        {
            "challenge": "toy",
            "model": "gpt-4o",
            "eval_mode": "insights_only",
            "with_skills": 0,
        }
    ]

# define the insights experiments
EXP_GROUPS["insights_prompts"] = []
for challenge in ["toy"]:
    for prompt_strategy in ["basic", "advanced"]:
        EXP_GROUPS["insights_prompts"] += [
            {
                "challenge": "toy",
                "model": "gpt-4o-mini",
                "eval_mode": "insights",
                "with_skills": 1,
                "prompt_strategy": prompt_strategy,
            }
        ]

# define the skills experiments
EXP_GROUPS["skills"] = []
for challenge in ["full"]:
    EXP_GROUPS["skills"] += [
        {"challenge": "full", "model": "gpt-4o-mini", "eval_mode": "skills"}
    ]

# define the models experiments

# Check for duplicates in each experiment group
for group_name, experiments in EXP_GROUPS.items():
    seen_configs = set()
    duplicates = []

    for exp in experiments:
        # Convert dict to a hashable format (frozen set of items)
        exp_tuple = frozenset(exp.items())

        if exp_tuple in seen_configs:
            duplicates.append(dict(exp_tuple))
        else:
            seen_configs.add(exp_tuple)

    if duplicates:
        raise ValueError(
            f"Duplicate configurations found in {group_name}: {duplicates}"
        )
