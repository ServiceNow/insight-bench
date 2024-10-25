import os, argparse
import pandas as pd

from insightbench.utils import agent_utils as au
from insightbench import agents, benchmarks
from insightbench.utils import exp_utils as eu
from insightbench.utils.exp_utils import hash_dict, save_json


def main(exp_dict, savedir, args):
    # Hyperparameters: Print Exp dict as hyperparamters and savedir
    print("\nExperiment Dict:")
    eu.print(exp_dict)
    print(f"\nSavedir: {savedir}\n")

    if args.reset and os.path.exists(savedir) and not args.eval_only:
        # assert savedir has exp_dict.json for safety
        assert os.path.exists(os.path.join(savedir, "exp_dict.json"))
        os.system(f"rm -rf {savedir}")

    save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    # Get Benchmark
    benchmark = benchmarks.get_benchmark(exp_dict["benchmark_type"])

    dataset_csv_path = os.path.join(datadir_path, dataset_csv_path)
    df = pd.read_csv(dataset_csv_path)

    if user_dataset_csv_path is not None:
        user_dataset_csv_path = os.path.join(datadir_path, user_dataset_csv_path)
        df_user = pd.read_csv(user_dataset_csv_path)
    else:
        df_user = None

    # load dataset

    for notebook_path in notebook_paths:
        agent = agents.Agent(
            dataset_csv_path=dataset_csv_path,
            user_dataset_csv_path=user_dataset_csv_path,
            model="gpt-4o",
            max_questions=exp_dict["max_questions"],
            branch_depth=exp_dict["branch_depth"],
            n_retries=2,
            savedir=savedir,
        )

        # Predict Insights
        pred_insights = agent.run_agent()

        # Get Notebook Evaluation
        scores_dict = {}
        for score_method in exp_dict["eval_metrics"]:
            pred_summary = agent.agent_poirot.summarize()
            pred_insights = [o["answer"] for o in agent.agent_poirot.insights_history]
            # compute summary score
            score_summary, summary_dict = agent.evaluate_agent_on_summary(
                gt_insights_dict, score_name=score_method, return_summary=True
            )

            score, score_dict = agent.evaluate_agent_on_notebook(
                gt_insights_dict, score_method=score_method
            )
            scores_dict[score_method + "_summary"] = summary_dict
            scores_dict[score_method + "_insights"] = score
            scores_dict[score_method + "_score_dict"] = score_dict

        # Save Scores
        save_json(os.path.join(savedir, "scores.json"), scores_dict)

        # Print Scores
        eu.print(scores_dict)

    print("Experiment Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sb", "--savedir_base", type=str, default="results")
    parser.add_argument("-r", "--reset", type=int, default=0)
    # add openai api key
    parser.add_argument("-o", "--openai_api_key", type=str, default=None)
    # dataset path
    parser.add_argument("-dp", "--dataset_path", type=str, default="data/notebooks")

    args, unknown = parser.parse_known_args()

    # exp_list
    exp_list = []
    for benchmark_type in ["toy"]:
        exp_list.append({"benchmark_type": benchmark_type})

    # Loop through experiments
    for exp_dict in exp_list:
        hash_id = hash_dict(exp_dict)
        savedir = os.path.join(args.savedir_base, hash_id)

        main(exp_dict, savedir, args)
