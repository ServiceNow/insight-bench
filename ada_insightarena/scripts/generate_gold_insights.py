import json
from typing import Dict, List
import os

from src.utils import get_llm_response


def load_qa_data(file_path: str) -> List[Dict]:
    """Load questions and answers from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_qa_for_prompt(qa_data: List[Dict]) -> str:
    """Format Q&A data into a string for the prompt."""
    formatted_data = []
    for item in qa_data:
        formatted_data.append(
            f"Question: {item['question']}\n" f"Answer: {item['answer']}"
        )
    return "\n\n".join(formatted_data)


def generate_insight(qa_data: List[Dict]) -> str:
    """Generate insights using LLM."""
    formatted_data = format_qa_for_prompt(qa_data)

    prompt = f"""Below is a collection of questions and answers about a dataset:

{formatted_data}

Based solely on the information available in these questions and answers, provide exactly 2 lines of the most informative insights. These insights can be about the dataset characteristics or any analysis/predictions made on the data. Ensure your insights are factual and directly supported by the Q&A pairs provided."""

    response = get_llm_response(prompt)

    return response


def main():
    # Path to your questions.json files
    data_path = "data/jsons"

    # Walk through all subdirectories in data/jsons
    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            questions_file = os.path.join(subdir_path, 'questions.json')

            print(f"\nProcessing: {subdir_path}")

            try:
                # Load the data
                qa_data = load_qa_data(questions_file)

                # Generate insight
                insight = generate_insight(qa_data)

                # Create output filename based on input path
                output_path = os.path.join(subdir_path, 'insights.json')

                # Print insights
                print("\nGenerated Insight:")
                print(insight)

                # Save insights to a file in the same directory as questions.json
                insight_dict = {
                    "insight": insight
                }
                with open(output_path, "w") as f:
                    json.dump(insight_dict, f, indent=4)

            except Exception as e:
                print(f"An error occurred processing {subdir_path}: {str(e)}")


if __name__ == "__main__":
    main()
