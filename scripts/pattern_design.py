import pandas as pd
import json
from openai import OpenAI
from typing import Dict, List
import os, re


class PatternDesigner:
    def __init__(self, api_key: str = None):
        """Initialize the PatternDesigner with OpenAI API key.

        Args:
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        self.client = OpenAI(api_key=api_key)

    def analyze_data(self, data: pd.DataFrame) -> str:
        """Analyze the data and return a detailed summary of its structure."""
        summary = {
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "column_summaries": {},
        }

        for col in data.columns:
            col_data = data[col]
            col_summary = {
                "dtype": str(col_data.dtype),
                "num_missing": int(col_data.isnull().sum()),
                "num_unique": int(col_data.nunique()),
                "sample_values": col_data.dropna().unique()[:3].tolist(),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_summary.update(
                    {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }
                )
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_summary.update(
                    {
                        "min_date": str(col_data.min()),
                        "max_date": str(col_data.max()),
                    }
                )
            elif pd.api.types.is_string_dtype(col_data):
                col_summary.update(
                    {"top_frequent_values": col_data.value_counts().head(3).to_dict()}
                )

            summary["column_summaries"][col] = col_summary

        return json.dumps(summary, indent=2)

    def design_patterns(self, data: pd.DataFrame, task: str) -> Dict[str, List[Dict]]:
        """Design patterns for each column based on the given analytics task."""
        data_summary = self.analyze_data(data)

        prompt = f"""
You are a data scientist creating a synthetic benchmark for a data analytics task. 
You are given a summary of a dataset and an analytics task. 
Your goal is to reason step-by-step and identify realistic patterns that can be injected into the data 
to make it more suitable for evaluating the performance of models on this task.

Please follow these steps for each column in the dataset:
1. Think about what kind of information this column conveys.
2. Consider how this column might affect or be related to the analytics task.
3. Suggest 1–2 *practical and realistic* patterns that could be injected into the column values.
4. Explain why injecting this pattern would help in evaluating the task.
5. Describe how the pattern would influence model learning or performance on the analytics task.

Use a JSON output format with the following structure:

{{
  "column_name_1": [
    {{
      "pattern": "description of the pattern",
      "reasoning": "explanation of why this pattern is useful",
      "relevance_to_task": "how this pattern helps with the task"
    }},
    ...
  ],
  ...
}}

Data Summary:
{data_summary}

Analytics Task:
{task}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o (OpenAI Omni)
            messages=[
                {
                    "role": "system",
                    "content": "You are a data pattern design expert. Your task is to suggest meaningful patterns that can be injected into data columns to help accomplish specific analytics tasks. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw_response = response.choices[0].message.content
        # Strip triple backticks and optional 'json' tag
        cleaned_json_str = re.sub(r"^```(?:json)?\n|\n```$", "", raw_response.strip())
        try:
            return json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")


def main():
    # Get API key from environment variable
    designer = (
        PatternDesigner()
    )  # Will automatically use OPENAI_API_KEY from environment

    # Sample DataFrame
    data = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "sales": [100, 150, 200],
            "category": ["A", "B", "A"],
        }
    )

    task = "Anomaly detection"

    try:
        patterns = designer.design_patterns(data, task)

        print("\nSuggested Patterns for Each Column:")
        for column, suggestions in patterns.items():
            print(f"\n{column}:")
            for suggestion in suggestions:
                print(f"\nPattern: {suggestion['pattern']}")
                print(f"Reasoning: {suggestion['reasoning']}")
                print(f"Relevance to task: {suggestion['relevance_to_task']}")
                print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
