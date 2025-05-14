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
        num_questions = 3

        prompt = f"""
        You are a data-centric AI expert designing synthetic data benchmarks to evaluate the reasoning ability of analytics models and agents.

        Given a {data_summary} and {task}, your job is to design **2–3 realistic, global patterns** that alter the dataset in meaningful ways. These patterns should:
        - Mimic plausible real-world behaviors or anomalies
        - Interact with the dataset's structure and semantics
        - Affect how key analytical questions are answered
        - Enable robust benchmarking of reasoning and insight extraction

        To ensure rigor and alignment, proceed through the following reasoning steps:

        ---

        ## 🔍 Phase 1: Identify Key Performance Indicators (KPIs)
        - Based on the dataset and task, identify 2–4 relevant KPIs that would be tracked by an analyst or model.
        - For each KPI:
          - Name the KPI
          - Explain what it measures
          - Identify the specific columns involved in its computation

        ---

        ## 🧠 Phase 2: Generate Analytical Questions (No Answers Yet)
        - Given the data summary and KPI, generate {num_questions} specific, quantitative data analytics questions that could be answered using this dataset. Focus on questions related to the {task}. 
        - Questions must require non-trivial reasoning (e.g., aggregation, filtering, correlations, conditional logic).
        - Do **not** answer them yet.
        - For each question, specify:
          - The question text
          - The KPI it targets
          - Which columns are needed to answer it

        ---

        ## 🧩 Phase 3: Analyze Column Interactions
        - Analyze how the KPIs and questions depend on the dataset’s columns.
        - Identify which columns are:
          - Most influential in computing KPIs
          - Likely to change the answer to multiple questions if perturbed
          - Involved in any cross-feature interactions, temporal dynamics, or category splits

        ---

        ## 🔬 Phase 4: Design 2–3 Realistic Global Patterns
        Each pattern should:
        - Involve 1 or more of the identified columns
        - Alter one or more KPIs in a way that causes at least one question's answer to change
        - simulate a **plausible real-world event, behavior, or trend**, such as:
          - A delayed effect (e.g., impact of feature A appears later in time)
          - A conditional behavior (e.g., sales increase only for category X on weekends)
          - A regime shift (e.g., before and after policy change)
          - A hidden dependency (e.g., correlations between latent group and outcome)
          - cross-feature dependencies
        - regime shifts
        - temporal or category-driven anomalies

        ### Phase 5: Explain for Each Pattern:
        - What exactly is the injected pattern?
        - Why is it useful from a benchmarking or insight perspective?
        - Which KPIs does it affect, and how?
        - What kind of analytical or modeling challenges does it test?
        - Which questions it changes and why


        ## 🧾 Phase 6: Post-Injection Answers
        - Now assume the dataset has been modified using the patterns above.
        - For each previously generated question, provide a **concise, numerically precise answer** based on the modified data.
        - Your answer should include:
          - Concrete values (e.g., counts, averages, rates, percentages) rather than vague statements
        - For each answer, also indicate:
          - Which pattern(s) caused the change
          - How the injected data behavior led to this specific numeric result


        ---

        ## 💡 Output Format (JSON)

        ```json
        {{
          "ID": "Unique identifier for the pattern",
          "data_summary": {data_summary},
          "Analytics task": {task},
          "kpis": [
            {{
              "name": "KPI name",
              "description": "What it measures and which columns are used"
            }}
          ],
          "questions": [
            {{
              "Question_index": "Index of the question. eg.Q1,Q2 etc.",
              "kpi": "KPI name",
              "question": "A natural language analytical question",
              "columns_required": ["list of columns"]
            }}
          ],
          "patterns": [
            {{
              "pattern_index": "Index of the pattern. eg.P1,P2 etc.",
              "pattern": "Description of the injected pattern",
              "columns_involved": ["list of columns affected"],
              "reasoning": "Why it's realistic and useful for benchmarking",
              "relevance_to_kpi": "Which KPIs it affects and how",
              "benchmark_value": "What kind of insight or model evaluation this pattern enables"
              "qa_impact": [
                {{
                  "question": "Impacted question",
                  "impact": "How the pattern changes the answer and why"
                }}
              ]
            }}
          ],
          "answers": [
            {{
              "question": "Question text",
              "answer_after_injection": "Correct answer based on injected data",
              "caused_by_pattern": "Index or name of the pattern that caused the change"
            }}
          ]
        }}

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
