#!/usr/bin/env python3
"""
This script aggregates human evaluation JSON files and computes, for each of 6 rubrics,
the percentage of outcomes in 4 categories:
  - insights_w_skills_pilot2ndrun
  - insights_wo_skills_pilot2ndrun
  - tie
  - none

For each JSON file, it:
  1. Extracts dataset_id, question_idx, and timestamp.
  2. Determines model assignment: if model_a.exp_group is "insights_w_skills_pilot2ndrun", then model_a wins for "A is better".
  3. For each rubric in the JSON’s "rubrics" section, it examines the "selection" field:
       - If selection == "A is better": outcome is model_a.exp_group’s value.
       - If selection == "B is better": outcome is model_b.exp_group’s value.
       - If selection == "Tie": outcome = "tie".
       - If selection == "None are good": outcome = "none".
  4. Saves this info in a DataFrame and writes it to CSV.
  5. Computes summary percentages for each rubric.
  
Usage:
    python compute_rubric_percentages.py <path_to_human_eval_jsons> [output_csv]
"""

import os
import sys
import json
import pandas as pd

def analyze_human_eval_files(folder_path, output_csv=None):
    data_rows = []

    # Iterate over all JSON files in the specified folder
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {file_name}: error reading JSON ({e})")
            continue

        # Skip if exp_group is null for either model
        model_a_group = data.get("model_a", {}).get("exp_group", None)
        model_b_group = data.get("model_b", {}).get("exp_group", None)
        if model_a_group is None or model_b_group is None:
            print(f"Skipping {file_name}: model_a or model_b exp_group is null.")
            continue

        # Basic info
        base_info = {
            "dataset_id": data.get("dataset_id"),
            "question_idx": data.get("question_idx"),
            "timestamp": data.get("timestamp")
        }

        # Determine model assignment (for reference)
        # If model_a.exp_group equals "insights_w_skills_pilot2ndrun", then model_a is "A"
        if model_a_group == "insights_w_skills_pilot2ndrun":
            base_info["insights_w_skills"] = "A"
            base_info["insights_wo_skills"] = "B"
        else:
            base_info["insights_w_skills"] = "B"
            base_info["insights_wo_skills"] = "A"

        # Process each rubric (we expect 6 rubrics)
        rubrics = data.get("rubrics", {})
        # We'll store both the raw selection and our computed outcome.
        for rubric_name, rubric_data in rubrics.items():
            selection = rubric_data.get("selection", None)
            outcome = None
            if selection == "A is better":
                # Use model_a's exp_group to decide outcome
                if model_a_group == "insights_w_skills_pilot2ndrun":
                    outcome = "insights_w_skills_pilot2ndrun"
                elif model_a_group == "insights_wo_skills_pilot2ndrun":
                    outcome = "insights_wo_skills_pilot2ndrun"
            elif selection == "B is better":
                # Use model_b's exp_group
                if model_b_group == "insights_w_skills_pilot2ndrun":
                    outcome = "insights_w_skills_pilot2ndrun"
                elif model_b_group == "insights_wo_skills_pilot2ndrun":
                    outcome = "insights_wo_skills_pilot2ndrun"
            elif selection == "Tie":
                outcome = "tie"
            elif selection == "None are good":
                outcome = "none"
            # Save both raw selection and computed outcome
            base_info[f"{rubric_name}_selection"] = selection
            base_info[f"{rubric_name}_outcome"] = outcome

        data_rows.append(base_info)

    # Create DataFrame from all evaluations
    df = pd.DataFrame(data_rows)

    # Optionally, save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")

    return df

def compute_summary(df):
    rubric_names = [
        "depth_of_analysis",
        "relevance_to_goal",
        "persona_consistency",
        "coherence",
        "answers_question_adequately",
        "plot_conclusion",
    ]

    print("\nDetailed Rubric Analysis:")
    print("=" * 50)
    for rn in rubric_names:
        col = f"{rn}_outcome"
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        total = counts.sum()
        print(f"\nRubric: {rn}")
        for outcome in ["insights_w_skills_pilot2ndrun", "insights_wo_skills_pilot2ndrun", "tie", "none"]:
            count = counts.get(outcome, 0)
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {outcome}: {count} ({pct:.2f}%)")
    print("=" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_rubric_percentages.py <path_to_human_eval_jsons> [output_csv]")
        sys.exit(1)
    folder_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "human_eval_results.csv"
    
    df = analyze_human_eval_files(folder_path, output_csv)
    print("\nAll evaluations:")
    print(df.head())
    
    compute_summary(df)

if __name__ == "__main__":
    main()

