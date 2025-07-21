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

    def design_patterns(
        self, data: pd.DataFrame, task: str, skills: List[str]
    ) -> Dict[str, List[Dict]]:
        """Design patterns for each column based on the given analytics task."""
        data_summary = self.analyze_data(data)
        num_questions = 10

        prompt = f"""
        You are a data-centric AI expert designing synthetic data benchmarks to evaluate the reasoning ability of analytics models and agents. Your goal is to design **5** diverse, realistic, global data patterns that can be injected into a dataset to rigorously evaluate reasoning and insight capabilities.

        You are provided with:
        {data_summary} – a high-level overview of the dataset.
        {task} – the analytical goal or task to be performed.
        {skills} – a list of analytical skills required to solve the task.
       These patterns should:
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

        ## 🧠 Phase 2: Generate Analytical Questions (Skill-Guided, No Answers Yet)
        You are given:
        - A high-level summary of the dataset
        - A list of  Key Performance Indicators (KPIs)
        - An analytics task: **`{task}`** (e.g., time series forecasting, anomaly detection, root cause analysis)
        - A list of relevant skills and algorithms: {skills} (e.g., ARIMA, k-means, XGBoost, Granger causality)

        Your job is to generate {num_questions} specific, quantitative, and algorithmically sophisticated analytical questions that meet the following criteria:

        Each question must:
        1. Be directly aligned with the task {task}
        2. Target a specific KPI from the provided list
        3. Be non-trivial — requiring more than basic group-by or filtering
        4. Be answerable using one or more algorithms from the {skills} list. Each question must implicitly or explicitly necessitate the use of that skill.

        ### For each question, return:
        - The **question text**
        - The **target KPI**
        - The **required columns** (from the dataset schema)
        - Associated skill/algorithm used
        - Task alignment explanation (how it relates to {task})
        - Algorithmic reasoning note (why this skill is appropriate or necessary)

        ---

        ## 🧩 Phase 3: Analyze Column Interactions
        - Analyze how the KPIs and questions depend on the dataset’s columns.
        - Identify which columns are:
          - Most influential in computing KPIs
          - Likely to change the answer to multiple questions if perturbed
          - Involved in any cross-feature interactions, temporal dynamics, or category splits

        ---

        ## 🔬 Phase 4: Design **5** Realistic Global Patterns
        Your task is to design **5 realistic data-level patterns or behaviors** to inject into the dataset.

        Each pattern must:

        - Involve **one or more of the key columns** identified in Phase 3 (especially those influencing KPIs and questions)
        - Be directly **relevant to the analytics task `{task}`** (e.g., trend shifts for forecasting, conditional breaks for anomaly detection, etc.)
        - **Change the answer to at least one of the questions** generated in Phase 2
        - Be detectable or analyzable using at least one algorithm from {skills}
        - Simulate a **plausible, real-world phenomenon**, such as:
          - **Delayed effects** (e.g., KPI changes occur with lag)
          - **Conditional behavior** (e.g., a trend only applies under certain filters)
          - **Regime shifts** (e.g., business logic or policy change at a time point)
          - **Hidden group behavior** (e.g., latent segment drives outcomes)
          - **Cross-feature interactions** (e.g., product type and time jointly affect behavior)
          - **Anomalies** (e.g., outliers that break expected patterns)

        Ensure that each pattern is **realistic and data-acquirable** (it must look like something that could exist in a real dataset).


        ### Phase 5: Explain for Each Pattern:
        - What exactly is the injected pattern?
        - Why is it useful from a benchmarking or insight perspective?
        - Which KPIs does it affect, and how?
        - What kind of analytical or modeling challenges does it test?
        - Which questions it changes and why


        ## 📊 Phase 6: Generate Insightful, Quantitative Answers Based on Modified Data

        Assume the dataset has been modified using the patterns injected in Phase 4.

        Your task is to revisit each of the previously generated questions and provide a **concise, natural-language answer** based solely on the modified dataset. The answers must sound like quantitative insights produced by a skilled analyst.

        ### 🔹 Direct Answer Requirements:
        - Express the answer as a **complete, natural-language insight** (not a list or number alone)
        - Phrase the answer as if explaining to a business stakeholder
        - Base the answer ** entirely on the modified dataset** (i.e., do not refer to any "change", "increase", or comparison to pre-injection values unless explicitly asked)
        - Includes **precise numerical values** (e.g., percentages, averages, counts, correlations) wherever possible
        - Avoids vague statements (e.g., "some improvement", "higher than usual")
        - Each answer should implicitly require or reflect application of a skill from the provided list (e.g., regression, clustering, causal inference)

      ### 🔸 Make the Pattern Effect Visibly Dominant:

      - Choose **extreme, yet realistic** values that make the effect of the injected pattern highly visible and unambiguous.
      - For example, if the expected rate could plausibly be 20%, modify the data so the answer becomes something like **60% or higher**, not just 22%
      - Ensure the values remain **plausible within the dataset's structure** (e.g., not breaking column ranges, types, or distributions completely).

        ### 🔹 Attribution and Explanation:
        - For each answer, also specify:
          - `patterns_triggered`: A list of injected pattern names that influenced the result
          - `explanation`: A brief 1–2 sentence summary of **how** the pattern affected the data and led to the specific insight

        ---

        ## 💡 Output Format (JSON)

        ```json
        {{
          "ID": "Unique identifier for the file",
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
              "caused_by_pattern": "Index of the pattern that caused the change"
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
