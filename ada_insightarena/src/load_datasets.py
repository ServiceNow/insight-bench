import pandas as pd
import json
import glob, os
from src.utils import check_and_fix_dataset


def get_dataset(challenge="toy"):
    """
    returns dataset as a list of dictionaries containing questions, metadata, goal, persona, insights, and table
    """
    base_path = "data/jsons"
    base_data_path = "data/csvs"

    # get the challenge
    print("challenge", challenge)
    if challenge == "toy":
        id_list = list(range(64,67))
    elif challenge == "pilot":
        # id_list = list(range(70,75))
        id_list= [346,447,482,551,700]
    elif challenge == "full":
        id_list = list(range(1, 26)) + list(range(27, 101))
    else:
        raise ValueError(f"Challenge {challenge} not found")

    print(id_list)
    data_list = []
    for _id in id_list:
        meta_dict = json.load(open(f"{base_path}/{_id}/meta.json", "r"))
        goal_dict = json.load(open(f"{base_path}/{_id}/goal.json", "r"))
        question_list = json.load(open(f"{base_path}/{_id}/questions.json", "r"))
        # insight_dict = json.load(open(f"{base_path}/{_id}/insights.json", "r"))

        data_dict = {}
        data_dict["id"] = _id
        data_dict["questions"] = question_list
        data_dict["dataset_path"] = f"{base_data_path}/{_id}/data.csv"
        data_dict["meta"] = meta_dict
        data_dict["goal"] = goal_dict["goal"]
        data_dict["persona"] = goal_dict["persona"]
        # data_dict["insight"] = insight_dict["insight"]
        data_dict["table"] = check_and_fix_dataset(f"{base_data_path}/{_id}/data.csv")

        reference_files_path = []
        for help_data_file_name in meta_dict["help_data_file_names"]:
            reference_files_path.append(f"{base_data_path}/{_id}/{help_data_file_name}")

        data_dict["reference_files"] = reference_files_path

        data_list.append(data_dict)

    return data_list


def load_dataset_insightarena(directory: str):
    """
    Reads all JSON files matching '*_*_patterns.json' in the given directory
    using glob.glob, and returns a dict mapping filename to parsed JSON content.
    """
    # pattern = os.path.join(directory, "*_*_patterns.json")
    file_paths = glob.glob(f"{directory}/jsons/*_*_patterns.json")
    # print(file_paths)

    data_list = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(data.keys())
        answers_questions = data['answers']
        task_name = "_".join(os.path.basename(file_path).split("_")[:2])
        print(task_name)
        for aq in answers_questions:
            data_dict = {}
            question = aq['question']
            pattern = aq['caused_by_pattern'].lower().replace(" ", "_")
            print(aq)
            print(pattern)
            data_dict['meta'] = {"dataset_description": data['data_summary']}
            data_dict['goal'] = f"The analysis aims to uncover {data['task']} insights from the given dataset"
            data_dict['persona']  = "You are a data analysis agent interesting in leveraging these insights in the real world"
            data_dict['questions'] = [{"question":aq["question"], "task":data['task']}] 
            data_dict['answers'] = [aq["answer_after_injection"]] 


            data_dict["dataset_path"] = f"{directory}/csvs/{task_name}_{pattern}_injected.csv"
            data_dict["table"] = check_and_fix_dataset(data_dict["dataset_path"])
            data_dict["id"] = f"{task_name}_{pattern}"
            data_list.append(data_dict)

    return data_list