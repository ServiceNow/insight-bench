import os, json
from dotenv import load_dotenv
from pprint import pprint
import pandas as pd
import exp_configs

# Load environment variables at the start
load_dotenv()

from src import load_datasets
from src import agents
from src import utils

import nltk

nltk.download("punkt_tab")

def main_insightarena(exp_dict, savedir, save_dir_categories, reset=False):
    print("Experiment:")
    print(exp_dict)
    # save it into savedir
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "exp_dict.json"), "w") as f:
        json.dump(exp_dict, f, indent=4)
    print("Experiment saved at: ", savedir)
    data_path = exp_dict['data_path']

    data_list = load_datasets.load_dataset_insightarena(data_path)

    for i, data_dict in enumerate(data_list):
        print(f"Working on experiment {i+1}/{len(data_list)}")
        score_dict = {}

        # Load the agent
        agent = agents.Agent(
            goal=data_dict["goal"],
            persona=data_dict["persona"],
            model=exp_dict["model"],
            data_description=data_dict["meta"]["dataset_description"],
            dataset_id=data_dict["id"]
        )

        if exp_dict["eval_mode"] == "insights":
            ## STAGE 2.2: Predict the Insights
            ## ==============================
            # get the prediction
            savedir_data = os.path.join(savedir, data_dict["id"])
            savedir_categories = os.path.join(
                save_dir_categories, data_dict["id"])
            
            os.makedirs(savedir_data, exist_ok=True)
            os.makedirs(savedir_categories, exist_ok=True)
            pred_insights = agent.predict_insights_wo_questions(
                table=data_dict["table"],
                questions=data_dict["questions"],
                savedir=savedir_data,
                savedir_categories=savedir_categories,
                skill_flag=exp_dict["with_skills"],
            )
            print(pred_insights)
            # get the ground truth
            gt_insights = data_dict['answers'][0]

            # visualize the insights
            agent.vis_insights(
                pred_insights=pred_insights,
                gt_insights=gt_insights,
                data_dict=data_dict,
                savedir=os.path.join(savedir, f"vis_{data_dict['id']}"),
            )



if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run experiments with specified experiment group"
    )
    parser.add_argument(
        "--exp_group",
        "-e",
        type=str,
        default="insights",
        choices=exp_configs.EXP_GROUPS.keys(),
        help="Experiment group to run from exp_configs.EXP_GROUPS",
    )
    args = parser.parse_args()

    # Load the experiments from the specified group
    exp_list = exp_configs.EXP_GROUPS[args.exp_group]

    print("\n\nExperiment group: ", args.exp_group)
    print("Number of experiments: ", len(exp_list))

    print("\n----------------------------------------\n")

    # run the experiments
    for exp_dict in exp_list:
        exp_hash = utils.get_exp_hash(exp_dict)
        savedir = f"results/{args.exp_group}/{exp_hash}"
        save_dir_categories = f"results/categories"
        main_insightarena(
            exp_dict,
            savedir=savedir,
            save_dir_categories=save_dir_categories,
            reset=True,
        )
