import json
import pandas as pd
from pathlib import Path


def analyze_human_eval_files(folder_path, output_csv=None):
    # List to store all the data
    data_rows = []

    # Iterate through all json files in the folder
    for file_path in Path(folder_path).glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)

            # Extract basic information
            base_info = {
                "dataset_id": data["dataset_id"],
                "question_idx": data["question_idx"],
                "timestamp": data["timestamp"],
            }

            # Determine which model is which
            if data["model_a"]["exp_group"] == "insights_w_skills_pilot":
                base_info["insights_w_skills"] = "A"
                base_info["insights_wo_skills"] = "B"
            else:
                base_info["insights_w_skills"] = "B"
                base_info["insights_wo_skills"] = "A"

            # Extract only rubric selections, simplify to A/B
            for rubric_name, rubric_data in data["rubrics"].items():
                selection = "A" if rubric_data["selection"] == "A is better" else "B"
                base_info[f"{rubric_name}_selection"] = selection

            data_rows.append(base_info)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Sort by dataset_id, question_idx, and timestamp
    df = df.sort_values(["dataset_id", "question_idx", "timestamp"])

    # Save to CSV if output_csv path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")

    return df


# Example usage:
# df = analyze_human_eval_files('path/to/your/folder')
# print(df)


def main():
    # Get the DataFrame and save to CSV
    df = analyze_human_eval_files("human_eval_pilot/", "human_eval_results.csv")

    # View all entries
    print("\nAll evaluations:")
    print(df)

    # Get a specific summary
    summary = (
        df.groupby(["dataset_id", "question_idx"])
        .agg(
            {
                "insights_w_skills": "first",
                "insights_wo_skills": "first",
                "timestamp": "count",
            }
        )
        .reset_index()
    )
    summary = summary.rename(columns={"timestamp": "count"})

    print(
        "\nModel assignments and number of evaluations per dataset_id and question_idx:"
    )
    print(summary)

    # Create binary results table (1 if insights_w_skills won, 0 if insights_wo_skills won)
    binary_results = df[["dataset_id", "question_idx", "timestamp"]].copy()
    for column in df.columns:
        if column.endswith("_selection"):
            binary_results[column] = df.apply(
                lambda row: 1 if row[column] == row["insights_w_skills"] else 0, axis=1
            )

    # Create a new DataFrame with summary rows
    final_results = []

    # Group by dataset_id and question_idx
    for (dataset_id, q_idx), group in binary_results.groupby(
        ["dataset_id", "question_idx"]
    ):
        # Add all individual evaluations first
        final_results.extend(group.to_dict("records"))

        # Calculate and add summary row
        summary_row = {
            "dataset_id": dataset_id,
            "question_idx": f"{dataset_id}_{q_idx}_summary",
            "timestamp": "insights_w_skills_score",
        }

        # Calculate mean score for each rubric
        for column in group.columns:
            if column.endswith("_selection"):
                summary_row[column] = round(group[column].mean(), 2)

        final_results.append(summary_row)

    # Convert to DataFrame
    final_df = pd.DataFrame(final_results)

    # Save to CSV
    final_df.to_csv("human_eval_results_binary.csv", index=False)
    print("\nResults with summary rows (decimal shows insights_w_skills winning rate):")
    print(final_df)

    # Calculate overall scores for each rubric
    print("\nDetailed Rubric Analysis:")
    print("=" * 50)

    rubric_names = {
        "depth_of_analysis_selection": "Depth of Analysis",
        "relevance_to_goal_selection": "Relevance to Goal",
        "persona_consistency_selection": "Persona Consistency",
        "coherence_selection": "Coherence",
        "answers_question_adequately_selection": "Answers Question Adequately",
        "plot_conclusion_selection": "Plot Conclusion",
    }

    total_w_skills = 0
    total_wo_skills = 0
    num_metrics = 0

    for column, display_name in rubric_names.items():
        if column in binary_results.columns:
            mean_score = round(binary_results[column].mean(), 2)
            total_w_skills += mean_score
            total_wo_skills += 1 - mean_score
            num_metrics += 1

            print(f"\n{display_name}:")
            print(f"insights_w_skills won:    {mean_score:.2f} ({mean_score*100:.0f}%)")
            print(
                f"insights_wo_skills won:   {1-mean_score:.2f} ({(1-mean_score)*100:.0f}%)"
            )

    # Calculate and display overall average
    if num_metrics > 0:
        overall_w_skills = round(total_w_skills / num_metrics, 2)
        overall_wo_skills = round(total_wo_skills / num_metrics, 2)
        print("\n" + "=" * 50)
        print(f"\nOVERALL AVERAGE:")
        print(
            f"insights_w_skills won:    {overall_w_skills:.2f} ({overall_w_skills*100:.0f}%)"
        )
        print(
            f"insights_wo_skills won:   {overall_wo_skills:.2f} ({overall_wo_skills*100:.0f}%)"
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
