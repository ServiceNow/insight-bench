# InsightBench Dataset Schema

This document describes the structure and conventions of the InsightBench dataset.

## Overview

InsightBench contains **97 flags** (analytical tasks) with a total of **409 ground-truth insights** for evaluating data analytics agents. Each flag consists of:
- A CSV dataset (simulated ServiceNow data)
- A Jupyter notebook with ground-truth analysis code and insights
- A JSON metadata file used by the evaluation framework

## JSON Schema

Each `flag-{N}.json` file has this structure:

```json
{
    "dataset_csv_path": "data/notebooks/csvs/flag-{N}.csv",
    "user_dataset_csv_path": "data/notebooks/csvs/flag-{N}-sysuser.csv | null",
    "metadata": {
        "goal": "Analytical goal description",
        "role": "Analyst role (e.g., 'L2 Support Agent')",
        "category": "One of the canonical categories",
        "dataset_description": "Description of the dataset fields",
        "header": "Task title",
        "difficulty": 1-4 (optional, null if not assigned)
    },
    "insight_list": [
        {
            "data_type": "descriptive | diagnostic | predictive | ...",
            "insight": "The key finding (ground truth)",
            "insight_value": { ... },
            "plot": { ... },
            "question": "The analytical question",
            "actionable_insight": "Recommended action",
            "code": "Python analysis code"
        }
    ],
    "insights": ["insight text 1", "insight text 2", ...],
    "summary": "Summary of all findings"
}
```

## Categories

| Category | Count | Description |
|----------|-------|-------------|
| Incident Management | 31 | ServiceNow incident analysis |
| Finance Management | 28 | Expense and financial analysis |
| Goal Management | 20 | Organizational goal tracking |
| User Management | 8 | HR and personnel analysis |
| Asset Management | 6 | IT asset management |
| Asset Management & User Management | 2 | Cross-domain analysis |
| Finance Management & User Management | 2 | Cross-domain analysis |

## Data Types (Insight Classification)

| Type | Count | Description |
|------|-------|-------------|
| descriptive | 188 | What happened? Distribution, averages, totals |
| diagnostic | 108 | Why did it happen? Correlations, root cause |
| comparative | 33 | How do groups differ? Cross-group analysis |
| predictive | 22 | What will happen? Trends, forecasts |
| analytical | 23 | Deep analysis of patterns and relationships |
| time_series | 17 | Temporal patterns and trends |
| correlative | 6 | Statistical correlations |
| trend diagnosis | 4 | Trend identification and analysis |
| frequency | 4 | Frequency and count analysis |
| distribution | 2 | Distribution shape analysis |
| prescriptive | 1 | What should be done? Recommendations |
| categorical | 1 | Categorical data analysis |

## Adversarial Flags (80-87)

Flags 80-87 are **deliberately adversarial** test cases. Their CSVs intentionally lack the columns required by the analysis. The ground-truth insights are error messages like "There was no column X to conduct any analysis."

These flags test whether data analytics agents:
1. Correctly identify when required data is unavailable
2. Report the limitation instead of hallucinating insights
3. Do not fabricate analysis results

## Difficulty Levels

Difficulty is assigned to 12 flags (from notebook metadata):

| Level | Flags |
|-------|-------|
| 1 | 12, 59 |
| 2 | 3, 49 |
| 3 | 2, 48 |
| 4 | 1, 47, 94, 95, 96, 97 |

## Evaluation

The evaluation framework uses the `insights` array (list of insight text strings) for scoring via ROUGE-1 or G-EVAL. The `insight_list` provides the full structured ground truth.

Benchmark tiers:
- `"toy"`: flags 1-5 (5 flags)
- `"standard"`: flags 1-30 (30 flags)
- `"full"`: flags 1-97 excluding duplicates (97 flags)

