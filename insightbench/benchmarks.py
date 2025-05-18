import os
import pandas as pd
from insightbench.utils.agent_utils import analysis_nb_to_gt

from evaluation.metrics_utils import score_insight
from evaluation import metrics
import nbformat, re, json


def get_benchmark(dataset_type, datadir):
    """load a set of tables and a set of golden insights to evaluate the agent's performance"""

    if dataset_type == "toy":
        flag_list = list(range(1, 6))
    elif dataset_type == "standard":
        flag_list = list(range(1, 31))
    elif dataset_type == "full":
        flag_list = list(range(1, 101))
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")

    return [f"{datadir}/flag-{flag}.json" for flag in flag_list]


def create_jsons(datadir):
    import glob

    datasets = glob.glob(f"{datadir}/flag-*.ipynb")
    for d in datasets:
        dataset_dict = extract_notebook_info(d)
        print("success:", d)


def load_dataset_dict(dataset_json_path):
    # load json
    with open(dataset_json_path, "r") as f:
        return json.load(f)


def extract_notebook_info(notebook_path):

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Initialize variables
    metadata = {
        "goal": None,
        "role": None,
        "category": None,
        "dataset_description": None,
        "header": None,
    }
    insight_list = []
    summary = None

    # Process each cell
    for cell in nb.cells:
        # Look for metadata in markdown cells containing "Dataset Description"
        if cell.cell_type == "markdown" and "Dataset Description" in cell.source:
            content = cell.source

            # More flexible patterns for all metadata fields
            goal_match = re.search(
                r"Goal[\s\*]*:[:\s]*(.*?)(?=\n\n|\n\*\*|\*\*|$)", content, re.IGNORECASE
            )
            if goal_match:
                metadata["goal"] = goal_match.group(1).strip()

            role_match = re.search(
                r"Role[\s\*]*:[:\s]*(.*?)(?=\n\n|\n\*\*|\*\*|$)", content, re.IGNORECASE
            )
            if role_match:
                metadata["role"] = role_match.group(1).strip()

            category_match = re.search(
                r"Category[\s\*]*:[:\s]*(.*?)(?=\n\n|\n\*\*|\*\*|$)",
                content,
                re.IGNORECASE,
            )
            if category_match:
                metadata["category"] = category_match.group(1).strip()

            # More flexible dataset description pattern
            desc_match = re.search(
                r"(?:###\s*Dataset Description|Dataset Description:?)\s*(.*?)(?=\n\n(?:\*\*|###)|$)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            if desc_match:
                metadata["dataset_description"] = desc_match.group(1).strip()

            # More flexible header pattern
            header_match = re.search(r"^(?:##\s*|)([^\n]+)", content)
            if header_match:
                metadata["header"] = header_match.group(1).strip()

        # Look for summary in markdown cells
        elif (
            cell.cell_type == "markdown"
            and "summary of findings".lower() in cell.source.lower()
        ):
            # Find the index case-insensitively but keep original text
            pattern = re.compile("summary of findings", re.IGNORECASE)
            match = pattern.search(cell.source)
            if match:
                summary = cell.source[match.end() :].strip()

        # Process code cells for insights
        elif cell.cell_type == "code" and cell.source.strip().startswith("{"):
            try:
                data = json.loads(cell.source)
                data["code"] = code
                if isinstance(data, dict):
                    insight_list.append(data)
            except Exception as e:
                # Extract insight and question directly from the text
                source = cell.source.strip()
                insight_match = re.search(r'"insight":\s*"([^"]*)"', source)
                question_match = re.search(r'"question":\s*"([^"]*)"', source)

                insight_dict = {
                    "insight": insight_match.group(1) if insight_match else "",
                    "question": question_match.group(1) if question_match else "",
                    "code": code,
                }
                insight_list.append(insight_dict)
        elif cell.cell_type == "code":
            code = cell.source.strip()

    # get flag id from notebook path
    flag = notebook_path.split("-")[-1].split(".")[0]
    # only if user_dataset_csv_path exists
    user_dataset_csv_path = (
        f"data/notebooks/csvs/flag-{flag}-sysuser.csv"
        if os.path.exists(f"data/notebooks/csvs/flag-{flag}-sysuser.csv")
        else None
    )
    dataset_dict = {
        "dataset_csv_path": f"data/notebooks/csvs/flag-{flag}.csv",
        "user_dataset_csv_path": user_dataset_csv_path,
        "metadata": metadata,
        "insight_list": insight_list,
        "insights": [ins["insight"] for ins in insight_list],
        "summary": summary[summary.find("\n") :],
    }
    assert os.path.exists(
        dataset_dict["dataset_csv_path"]
    ), f"Dataset path {dataset_dict['dataset_csv_path']} does not exist"
    # save dataset_dict to json
    with open(f"data/notebooks/flag-{flag}.json", "w") as f:
        json.dump(dataset_dict, f, indent=4)

    return dataset_dict


def evaluate_insights(pred_insights, gt_insights, score_name="rouge1"):
    # compute score using score_method
    if score_name == "rouge1":
        score, score_dict = metrics.compute_rouge(pred_insights, gt_insights)
    elif score_name == "g_eval":
        score, score_dict = metrics.compute_g_eval_o2m(pred_insights, gt_insights)

    return score


def evaluate_summary(pred, gt, score_name="rouge1"):
    score_summary = score_insight(
        pred_insight=pred, gt_insight=gt, score_name=score_name
    )

    return score_summary
