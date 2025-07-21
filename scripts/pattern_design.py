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
        You are a data-centric AI expert designing synthetic data benchmarks to evaluate analytics models.

        Given a dataset summary and an analytics task, your job is to inject **2–3 realistic patterns across one or more columns** that:
        - Mimic real-world behaviors or anomalies
        - Interact with the dataset's structure and semantics
        - Meaningfully impact model performance or insight extraction
        - Allow for robust benchmarking of analytical reasoning

        ---

        Please follow these explicit steps in your reasoning (Chain-of-Thought):

        ### Step 1: Infer Key Performance Indicators (KPIs)
        - Based on the dataset and task, identify 2–4 relevant KPIs that would be tracked by an analyst or model.

        ### Step 2: Identify Influential Columns and Relationships
        - Which columns most influence these KPIs?
        - Are there any natural correlations, temporal dynamics, or category-based splits that could affect KPI computation?

        ### Step 3: Design 2–3 Global Patterns
        - Each pattern may involve **1 or more columns**, and should simulate a **plausible real-world event, behavior, or trend**.
        - Avoid trivial noise (e.g., "random fluctuation"). Prefer **interpretable and benchmark-worthy** signals like:
        - delayed effects
        - conditionally induced trends
        - cross-feature dependencies
        - regime shifts
        - temporal or category-driven anomalies

        ### Step 4: Explain for Each Pattern:
        - What exactly is the injected pattern?
        - Why is it useful from a benchmarking or insight perspective?
        - Which KPIs does it affect, and how?
        - What kind of analytical or modeling challenges does it test?

        ---

        ### Output format (JSON):

        {{
        "kpis": ["list of important KPIs"],
        "patterns": [
            {{
            "pattern": "Description of the injected pattern",
            "columns_involved": ["list of columns affected"],
            "reasoning": "Why this pattern is meaningful and realistic",
            "relevance_to_kpi": "Which KPIs it affects and how",
            "benchmark_value": "What kind of insight or model evaluation this pattern enables"
            }},
            ...
        ]
        }}

        ---

        ### Data Summary:
        {data_summary}

        ### Analytics Task:
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

        print("\nKey Performance Indicators (KPIs):")
        for kpi in patterns.get("kpis", []):
            print(f"- {kpi}")

        print("\nSuggested Patterns:")
        for pattern in patterns.get("patterns", []):
            print(f"\nPattern: {pattern['pattern']}")
            print(f"Columns Involved: {', '.join(pattern['columns_involved'])}")
            print(f"Reasoning: {pattern['reasoning']}")
            print(f"Relevance to KPI: {pattern['relevance_to_kpi']}")
            print(f"Benchmark Value: {pattern['benchmark_value']}")
            print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
