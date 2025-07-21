import os
import pandas as pd
from pathlib import Path
import re

# Move score_categories to module level
score_categories = [
    "Depth of Analysis:",
    "Relevance to Goal:",
    "Persona Consistency:",
    "Coherence:",
    "Answers Question Adequately:",
    "Plot Conclusion:",
]


def extract_scores_from_section(lines):
    # Remove the score_categories definition from here since it's now at module level
    scores = []
    for line in lines:
        if any(category in line for category in score_categories):
            try:
                score = float(line.split(":")[1].split("/")[0].strip())
                scores.append(score)
            except (ValueError, IndexError):
                continue
    return sum(scores) / len(scores) if scores else 0


def analyze_evaluation_scores():
    # Path to evaluations directory
    eval_dir = Path("results/evaluations/GPTEval/pilot")
    # Dictionary to store scores by folder ID
    folder_scores = {}

    # Walk through all folders in evaluations directory
    for folder in eval_dir.iterdir():
        if folder.is_dir():
            folder_id = folder.name
            response_file = folder / "llm_response.txt"
            if response_file.exists():
                with open(response_file, "r") as f:
                    content = f.read()

                    # Dictionary to store scores for each category
                    with_skills_scores = {}
                    without_skills_scores = {}

                    # Process content by sections
                    lines = content.split("\n")
                    current_category = None
                    looking_for_insight1 = True
                    looking_for_insight2 = False

                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()

                        # First look for category
                        if any(category in line for category in score_categories):
                            current_category = next(
                                cat.rstrip(":")
                                for cat in score_categories
                                if cat in line
                            )
                            looking_for_insight1 = True
                            looking_for_insight2 = False

                        # Then look for Insight 1
                        elif (
                            current_category
                            and looking_for_insight1
                            and "Insight 1" in line
                        ):
                            decimal_match = re.search(r"\d+\.\d+", line)
                            if decimal_match:
                                with_skills_scores[current_category] = float(
                                    decimal_match.group()
                                )
                                looking_for_insight1 = False
                                looking_for_insight2 = True

                        # After finding Insight 1, look for Insight 2
                        elif (
                            current_category
                            and looking_for_insight2
                            and "Insight 2" in line
                        ):
                            decimal_match = re.search(r"\d+\.\d+", line)
                            if decimal_match:
                                without_skills_scores[current_category] = float(
                                    decimal_match.group()
                                )
                                looking_for_insight2 = False
                                # Reset category after finding both scores
                                current_category = None

                        i += 1

                    folder_scores[folder_id] = (
                        with_skills_scores,
                        without_skills_scores,
                    )

    # Create DataFrame with detailed scores
    data = []
    for folder_id, (with_scores, without_scores) in folder_scores.items():
        row = {"Folder ID": folder_id}
        for category in score_categories:
            category_name = category.rstrip(":")
            row[f"{category_name} (W)"] = with_scores.get(category_name, 0)
            row[f"{category_name} (WO)"] = without_scores.get(category_name, 0)
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values("Folder ID")

    # Save results to CSV
    df.to_csv("results/evaluations/GPTEval/pilot/evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

    # Display results
    print("\nEvaluation Results:")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.to_string(index=False))

    # Calculate and display overall averages
    print("\nOverall Averages:")
    for category in score_categories:
        category_name = category.rstrip(":")
        with_avg = df[f"{category_name} (W)"].mean()
        without_avg = df[f"{category_name} (WO)"].mean()
        print(
            f"{category_name:30}: With Skills {with_avg:.2f} | Without Skills {without_avg:.2f}"
        )
    print("=" * 100)

    return df


if __name__ == "__main__":
    analyze_evaluation_scores()
