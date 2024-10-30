import os, argparse
import pandas as pd

from insightbench.utils import agent_utils as au
from insightbench import agents, benchmarks
from insightbench.utils import exp_utils as eu
from insightbench.utils.exp_utils import hash_dict, save_json


def main(exp_dict, savedir, args):
    # Hyperparameters:
    # ----------------

    # Print Exp dict as hyperparamters and savedir
    print("\nExperiment Dict:")
    eu.print(exp_dict)
    print(f"\nSavedir: {savedir}\n")

    # Reset savedir if reset flag is set
    if args.reset and os.path.exists(savedir) and not args.eval_only:
        # assert savedir has exp_dict.json for safety
        assert os.path.exists(os.path.join(savedir, "exp_dict.json"))
        os.system(f"rm -rf {savedir}")

    save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    # Get Benchmark:
    # ----------------
    notebook_list = benchmarks.get_benchmark(
        exp_dict["benchmark_type"], datadir=args.datadir
    )

    # load agent
    agent = agents.Agent(
        model_name=exp_dict["model_name"],
        max_questions=exp_dict["max_questions"],
        branch_depth=exp_dict["branch_depth"],
        n_retries=2,
        savedir=savedir,
    )

    # load dataset
    score_list = []
    for notebook_dict in notebook_list:
        # Load Dataset
        dataset_dict = benchmarks.load_dataset_dict(
            dataset_csv_path=notebook_dict["dataset_csv_path"],
            dataset_notebook_path=notebook_dict["notebook_path"],
            user_dataset_csv_path=notebook_dict["user_dataset_csv_path"],
        )

        # Predict Insights
        pred_insights = agent.get_insights(
            dataset_csv_path=dataset_dict["dataset_csv_path"],
            user_dataset_csv_path=dataset_dict["user_dataset_csv_path"],
        )
        pred_summary = agent.agent_poirot.summarize(pred_insights)

        # Evaluate Agent
        # --------------
        gt_insights_dict = dataset_dict["gt_insights_dict"]
        gt_summary = gt_insights_dict["flag"]

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
    parser.add_argument("-d", "--datadir", type=str, default="data/notebooks")

    args, unknown = parser.parse_known_args()

    # exp_list
    exp_list = []
    for benchmark_type in ["toy"]:
        for model_name in ["gpt-4o-mini"]:
            exp_list.append(
                {
                    "benchmark_type": benchmark_type,
                    "model_name": model_name,
                    "max_questions": 2,
                    "branch_depth": 1,
                }
            )

    # set open ai env
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Loop through experiments
    for exp_dict in exp_list:
        hash_id = hash_dict(exp_dict)
        savedir = os.path.join(args.savedir_base, hash_id)

        main(exp_dict, savedir, args)
