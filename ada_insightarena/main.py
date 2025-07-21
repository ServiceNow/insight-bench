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


def main(exp_dict, savedir, save_dir_categories, reset=False):
    # Pretty print the experiment dictionary
    print("Experiment:")
    print(exp_dict)
    # save it into savedir
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "exp_dict.json"), "w") as f:
        json.dump(exp_dict, f, indent=4)
    print("Experiment saved at: ", savedir)

    ## STAGE 1: Load the Dataset
    ## ==============================
    # TODO: Load the data Amirhossein
    # the dataset is a list of dictionaries containing questions, metadata, goal, persona, insights, and table
    data_list = load_datasets.get_dataset(challenge=exp_dict["challenge"])

    score_list = []
    all_skill_scores = []
    for i, data_dict in enumerate(data_list):
        print(f"Working on experiment {i+1}/{len(data_list)}")
        score_dict = {}

        # Load the agent
        agent = agents.Agent(
            goal=data_dict["goal"],
            persona=data_dict["persona"],
            model=exp_dict["model"],
            data_description=data_dict["meta"]["dataset_description"],
            dataset_id=str(data_dict["id"])
        )

        if exp_dict["eval_mode"] == "skills":
            ## STAGE 2.1: Predict the Skills (Ablation)
            ## ==============================
            # predict the skills
            gt_questions = data_dict["questions"][:100]

            skills_list = agent.predict_skills(questions=gt_questions)

            # get the ground truth skills
            gt_skills = [
                q["skill"]
                .replace("Collaborative Filtering", "collaborativefiltering")
                .replace("Granger Causality", "GrangerCausality")
                .replace("Isolation Forest", "isolationforest")
                .replace("K-Means Clustering", "kmeans")
                .replace("KNN Imputation", "KNNImputation")
                .replace("Latent Dirichlet Allocation", "LDA")
                .replace("Multi-Armed Bandit", "MultiArmedBandit")
                .replace("Naive Bayes", "naivebayes")
                .replace("Neural Networks", "neuralnetworks")
                .replace("PageRank", "pagerank")
                .replace("Random Forest", "randomforest")
                .replace("RFM Analysis", "RFMAnalysis")
                .replace("Spearman Correlation", "SpearmanCorrelationCoefficient")
                .replace("SVD", "svd-nmf-topic-modelling")
                .replace("randomforest Importance", "RandomForestFeatureImportance")
                .replace("Kernel PCA", "Kernel_PCA")
                .replace("Eigenvalue Decomposition", "EigenDecomposition")
                .replace("Pearson Correlation", "PearsonCorrelation")
                .replace("Student's T-Test", "Ttest")
                for q in gt_questions
            ]

            pred_skills = [s["predicted_skills"] for s in skills_list]
            # score the skills
            skill_score = agent.score_skills(
                pred_skills=pred_skills, gt_skills=gt_skills, method="mrr"
            )

            score_dict["skill_score"] = skill_score["score"]
            score_dict["id"] = data_dict["id"]
            score_dict["goal"] = data_dict["goal"]
            score_dict["persona"] = data_dict["persona"]

            # Add to a list of scores for averaging
            all_skill_scores.append(skill_score["score"])

            # visualize the skills
            agent.vis_skills(
                pred_skills=pred_skills,
                gt_skills=gt_skills,
                questions=gt_questions,
                savedir=os.path.join(savedir, f"vis_{i}"),
            )

        elif exp_dict["eval_mode"] == "insights":
            ## STAGE 2.2: Predict the Insights
            ## ==============================
            # get the prediction
            savedir_data = os.path.join(savedir, str(data_dict["id"]))
            savedir_categories = os.path.join(
                save_dir_categories, str(int(data_dict["id"]) - 1)
            )
            os.makedirs(savedir_data, exist_ok=True)
            os.makedirs(savedir_categories, exist_ok=True)
            pred_insights = agent.predict_insights(
                table=data_dict["table"],
                savedir=savedir_data,
                savedir_categories=savedir_categories,
                skill_flag=exp_dict["with_skills"],
            )
            print(pred_insights)
            # get the ground truth
            gt_insights = ""

            # get the score
            # score_dict = agent.score_insights(
            #     pred_insights=pred_insights, gt_insights=gt_insights
            # )
            # score_dict["id"] = data_dict["id"]
            # score_dict["goal"] = data_dict["goal"]
            # score_dict["persona"] = data_dict["persona"]

            # visualize the insights
            agent.vis_insights(
                pred_insights=pred_insights,
                gt_insights=gt_insights,
                data_dict=data_dict,
                savedir=os.path.join(savedir, f"vis_{str(int(data_dict['id'])-1)}"),
            )

        elif exp_dict["eval_mode"] == "insights_only":
            ## STAGE 2.3: Predict the Insights based on gt questions
            ## ==============================
            # get the prediction
            gt_questions = data_dict["questions"]
            gt_insights = data_dict["insight"]
            gt_skills = [
                q["skill"]
                .replace("Collaborative Filtering", "collaborativefiltering")
                .replace("Granger Causality", "GrangerCausality")
                .replace("Isolation Forest", "isolationforest")
                .replace("K-Means Clustering", "kmeans")
                .replace("KNN Imputation", "KNNImputation")
                .replace("Latent Dirichlet Allocation", "LDA")
                .replace("Multi-Armed Bandit", "MultiArmedBandit")
                .replace("Naive Bayes", "naivebayes")
                .replace("Neural Networks", "neuralnetworks")
                .replace("PageRank", "pagerank")
                .replace("Random Forest", "randomforest")
                .replace("RFM Analysis", "RFMAnalysis")
                .replace("Spearman Correlation", "SpearmanCorrelationCoefficient")
                .replace("SVD", "svd-nmf-topic-modelling")
                for q in gt_questions
            ]
            # print(gt_skills)
            savedir_data = os.path.join(savedir, str(i))
            os.makedirs(savedir_data, exist_ok=True)
            pred_insights = agent.predict_insights_only(
                table=data_dict["table"],
                savedir=savedir_data,
                skill_flag=exp_dict["with_skills"],
                questions=gt_questions,
                skills=gt_skills,
            )
            # print(pred_insights)
            # break
            # get the ground truth

            # get the score
            score_dict = agent.score_insights(
                pred_insights=pred_insights, gt_insights=gt_insights
            )
            score_dict["id"] = data_dict["id"]
            score_dict["goal"] = data_dict["goal"]
            score_dict["persona"] = data_dict["persona"]

            # visualize the insights
            agent.vis_insights(
                pred_insights=pred_insights,
                gt_insights=gt_insights,
                data_dict=data_dict,
                savedir=os.path.join(savedir, f"vis_{i}"),
            )
        # break

        # save the score
        # score_list.append(score_dict)

        # # print the head of the score list
        # print(pd.DataFrame(score_list).tail())

        # # save the score list
        # score_df = pd.DataFrame(score_list)
        # score_df.to_csv(os.path.join(savedir, "score_list.csv"), index=False)

        print(f"\nExperiment {i+1}/{len(data_list)} updated in ", savedir)
        print("\n==============================================\n")

    # Calculate and print average skill score after the loop
    # average_skill_score = sum(all_skill_scores) / len(all_skill_scores)
    # print(f"Average skill score: {average_skill_score:.4f}")

    print("Experiment completed at savedir: ", savedir)
    print()

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
        savedir = f"insightarena/results/{args.exp_group}/{exp_hash}"
        save_dir_categories = f"insightarena/results/categories"
        main_insightarena(
            exp_dict,
            savedir=savedir,
            save_dir_categories=save_dir_categories,
            reset=True,
        )
