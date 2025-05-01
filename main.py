import os, argparse
import pandas as pd
from insightbench.utils import agent_utils as au
from insightbench import agents, benchmarks
from insightbench.utils import exp_utils as eu
from insightbench.utils.exp_utils import hash_dict, save_json
from dotenv import load_dotenv

load_dotenv()


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
    dataset_list = benchmarks.get_benchmark(
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
    for dataset_json_path in dataset_list:
        # Load Dataset
        dataset_dict = benchmarks.load_dataset_dict(dataset_json_path=dataset_json_path)

        # Predict Insights
        pred_insights, pred_summary = agent.get_insights(
            dataset_csv_path=dataset_dict["dataset_csv_path"],
            user_dataset_csv_path=dataset_dict["user_dataset_csv_path"],
        )
        # Evaluate Agent
        # --------------
        # Evaluate
        score_insights = benchmarks.evaluate_insights(
            pred_insights=pred_insights,
            gt_insights=dataset_dict["insights"],
            score_name="rouge1",
        )
        score_summary = benchmarks.evaluate_summary(
            pred=pred_summary, gt=dataset_dict["summary"], score_name="rouge1"
        )

        score_list.append(
            {
                "score_insights": score_insights,
                "score_summary": score_summary,
            }
        )
        # Print Scores
        print(pd.DataFrame(score_list).tail())

        # save score_list
        save_json(os.path.join(savedir, "score_list.json"), score_list)

    print("Experiment Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sb", "--savedir_base", type=str, default="results")
    parser.add_argument("-r", "--reset", type=int, default=0)
    # add openai api key
    # parser.add_argument("-o", "--openai_api_key", type=str, default=None)
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
    # os.environ["OPENAI_API_KEY"] =

    # Loop through experiments
    for exp_dict in exp_list:
        hash_id = hash_dict(exp_dict)
        savedir = os.path.join(args.savedir_base, hash_id)

        main(exp_dict, savedir, args)
