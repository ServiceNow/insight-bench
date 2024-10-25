import os, argparse
import scripts.exp_groups as exp_groups
import pandas as pd

from insightbench.utils import agent_utils as au
from insightbench import agents
from insightbench.utils import exp_utils as eu
from insightbench.utils.exp_utils import hash_dict, save_json


def main(exp_dict, savedir, args):
    # Print Exp dict as hyperparamters and savedir
    print("\nExperiment Dict:")
    eu.print(exp_dict)
    print(f"\nSavedir: {savedir}\n")

    if args.reset and os.path.exists(savedir) and not args.eval_only:
        # assert savedir has exp_dict.json for safety
        assert os.path.exists(os.path.join(savedir, "exp_dict.json"))
        os.system(f"rm -rf {savedir}")

    save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    # Get dataset paths
    user_dataset_csv_path = None
    datadir_path = "data/notebooks"
    if exp_dict["do_sensitivity"]:
        fname_ipynb = os.path.join(datadir_path, "flag-2.ipynb")
        gt_insights_dict = au.analysis_nb_to_gt(fname_ipynb)
        slope = exp_dict["slope"]
        dataset_csv_path = (
            f"csvs/sensitivity_analysis/incidents_with_slope_{slope:03d}.csv"
        )
    else:
        fname_ipynb = os.path.join(datadir_path, f"flag-{exp_dict['dataset_id']}.ipynb")
        gt_insights_dict = au.analysis_nb_to_gt(fname_ipynb)
        dataset_csv_path = gt_insights_dict["dataset_csv_path"]
        user_dataset_csv_path = gt_insights_dict["user_dataset_csv_path"]

    dataset_csv_path = os.path.join(datadir_path, dataset_csv_path)
    df = pd.read_csv(dataset_csv_path)

    if user_dataset_csv_path is not None:
        user_dataset_csv_path = os.path.join(datadir_path, user_dataset_csv_path)
        df_user = pd.read_csv(user_dataset_csv_path)
    else:
        df_user = None

    # Get goal
    if exp_dict["use_goal"]:
        goal = gt_insights_dict["goal"]
    else:
        goal = "I want to find interesting trends in this dataset"

    # Get Agent
    agent = agents.Agent(
        df=df,
        df_user=df_user,
        dataset_csv_path=dataset_csv_path,
        user_dataset_csv_path=user_dataset_csv_path,
        gen_engine=exp_dict["gen_engine"],
        goal=goal,
        max_questions=exp_dict["max_questions"],
        branch_depth=exp_dict["branch_depth"],
        n_retries=2,
        savedir=savedir,
        temperature=exp_dict.get("temperature", 0),
    )

    # Run Agent or Run Evaluation Only
    if not args.eval_only:
        # Run Full Data Analytics
        pred_insights_dict = agent.run_agent()
    else:
        # Load pred_insights_dict
        agent.load_checkpoint(savedir)

    # Get Notebook Evaluation
    scores_dict = {}
    for score_method in exp_dict["eval_metrics"]:
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
    parser.add_argument("--database_path", type=str, default="data/notebooks/csvs/")
    parser.add_argument("-sb", "--savedir_base", type=str, default="results")
    parser.add_argument("-r", "--reset", type=int, default=0)
    parser.add_argument("--eval_only", type=int, default=0)
    # add openai api key
    parser.add_argument("-o", "--openai_api_key", type=str, default=None)

    parser.add_argument(
        "-e",
        "--exp_group",
        default="quick",
        help="name of an experiment in exp_groups.py",
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=4)

    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, unknown = parser.parse_known_args()

    # Get Experiment Group
    exp_list = exp_groups.EXP_GROUPS[args.exp_group]

    # Loop through experiments
    for exp_dict in exp_list:
        hash_id = hash_dict(exp_dict)
        savedir = os.path.join(args.savedir_base, args.exp_group, hash_id)
        main(exp_dict, savedir, args)
