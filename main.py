import os, argparse
import pandas as pd
from insightbench.utils import agent_utils as au
from insightbench import agents, benchmarks
from insightbench.utils import exp_utils as eu
from insightbench.utils.exp_utils import hash_dict, save_json
from dotenv import load_dotenv
from insightbench.agents import AgentDataGen
import json

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

    # # Get Benchmark:
    # # ----------------
    # dataset_list = benchmarks.get_benchmark(
    #     exp_dict["benchmark_type"], datadir=args.datadir
    # )

    # load agent
    agent = agents.Agent(
        model_name=exp_dict["model_name"],
        max_questions=exp_dict["max_questions"],
        branch_depth=exp_dict["branch_depth"],
        n_retries=2,
        savedir=savedir,
    )

    # Load incident data from Excel
    incident_data = pd.read_excel("data/incident.xlsx", sheet_name="incident data")

    # load pattern generation agent
    pattern_agent = agents.AgentDataGen(
        api_key=os.getenv("OPENAI_API_KEY"),
        tasks_path="insightbench/utils/domains_tasks.json",
        dataset=incident_data,
    )
    # Convert timestamp columns to string format
    for col in incident_data.columns:
        if pd.api.types.is_datetime64_any_dtype(incident_data[col]):
            incident_data[col] = incident_data[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Generate patterns for all tasks
    pattern_agent.generate_all_patterns(
        data=incident_data,
        output_dir=os.path.join(savedir, "patterns"),
        hash_id=hash_id,
    )
    # Get all CSV files from IBExt directory
    ibext_dir = "data/IBExt"
    csv_files = [f for f in os.listdir(ibext_dir) if f.endswith("_injected.csv")]

    score_list = []
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file}")
        dataset_csv_path = os.path.join(ibext_dir, csv_file)

        # Get the corresponding pattern file
        pattern_file = csv_file.replace("_injected.csv", "_patterns.json")
        pattern_path = os.path.join("results", hash_id, "patterns/tasks", pattern_file)

        if os.path.exists(pattern_path):
            # Extract task name from pattern file
            task = pattern_file.replace("_patterns.json", "").replace("_", " ").title()

            # Load patterns as ground truth insights
            with open(pattern_path, "r") as f:
                patterns_data = json.load(f)

            # Extract patterns and reasoning as ground truth insights
            gt_insights = []
            for pattern in patterns_data.get("patterns", []):
                gt_insights.append(
                    {
                        "pattern": pattern.get("pattern", ""),
                        "reasoning": pattern.get("reasoning", ""),
                        "relevance": pattern.get("relevance_to_kpi", ""),
                        "file": csv_file,
                        "task": task,
                    }
                )

            pred_insights, pred_summary = agent.get_insights(
                dataset_csv_path=dataset_csv_path, user_dataset_csv_path=None, task=task
            )

            # Evaluate
            score_insights = benchmarks.evaluate_insights(
                pred_insights=pred_insights,
                gt_insights=gt_insights,
                score_name="rouge1",
            )
            score_summary = benchmarks.evaluate_summary(
                pred=pred_summary,
                gt="\n".join(
                    [
                        f"Pattern: {insight['pattern']}\nReasoning: {insight['reasoning']}\nRelevance: {insight['relevance']}"
                        for insight in gt_insights
                    ]
                ),
                score_name="rouge1",
            )

            score_list.append(
                {
                    "score_insights": score_insights,
                    "score_summary": score_summary,
                    "file": csv_file,
                }
            )
            # Print Scores
            print(pd.DataFrame(score_list).tail())

            # save score_list
            save_json(os.path.join(savedir, "score_list.json"), score_list)

            print(f"Finished processing {csv_file}")
        else:
            print(f"Warning: Pattern file not found for {csv_file}")

    print("Experiment Done!")


#     for dataset_json_path in dataset_list:
#         # Load Dataset
#         dataset_dict = benchmarks.load_dataset_dict(
#             dataset_json_path=dataset_json_path
#         )

#         # Predict Insights
#         pred_insights, pred_summary = agent.get_insights(
#             dataset_csv_path=dataset_csv_path,
#             user_dataset_csv_path=None,
#         )
#         # Evaluate Agent
#         # --------------
#         # Evaluate
#         score_insights = benchmarks.evaluate_insights(
#             pred_insights=pred_insights,
#             gt_insights=dataset_dict["insights"],
#             score_name="rouge1",
#         )
#         score_summary = benchmarks.evaluate_summary(
#             pred=pred_summary, gt=dataset_dict["summary"], score_name="rouge1"
#         )

#         score_list.append(
#             {
#                 "score_insights": score_insights,
#                 "score_summary": score_summary,
#                 "file": csv_file,
#             }
#         )
#         # Print Scores
#         print(pd.DataFrame(score_list).tail())

#         # save score_list
#         save_json(os.path.join(savedir, "score_list.json"), score_list)

#     print(f"Finished processing {csv_file}")

# print("Experiment Done!")


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
        for model_name in ["gpt-4o"]:
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

    process_ibext_files()
