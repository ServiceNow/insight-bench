import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import glob

# Load tasks and skills from domains_tasks.json
with open("insightbench/utils/domains_tasks.json", "r") as f:
    domains_tasks = json.load(f)
    available_tasks = domains_tasks["tasks"]
    skills_dict = domains_tasks["SKILLS"]


def get_excel_sheets(file_obj):
    """Get list of sheets from Excel file."""
    try:
        return pd.ExcelFile(file_obj).sheet_names
    except Exception:
        return []


def load_dataset(file_obj, sheet_name=None):
    """Load dataset from uploaded file."""
    if file_obj is None:
        return pd.DataFrame()

    try:
        # Try to detect file type and load accordingly
        file_name = file_obj.name.lower()
        if file_name.endswith(".csv"):
            return pd.read_csv(file_obj)
        elif file_name.endswith((".xls", ".xlsx")):
            if sheet_name:
                return pd.read_excel(file_obj, sheet_name=sheet_name)
            else:
                # If no sheet specified, return first sheet
                return pd.read_excel(file_obj, sheet_name=0)
        elif file_name.endswith(".json"):
            return pd.read_json(file_obj)
        else:
            return pd.DataFrame(
                {
                    "error": [
                        "Unsupported file format. Please upload CSV, Excel, or JSON files."
                    ]
                }
            )
    except Exception as e:
        return pd.DataFrame({"error": [f"Error loading file: {str(e)}"]})


def load_task_patterns(task_name):
    """Load patterns and KPIs for the selected task."""
    try:
        # Search for pattern files in all FinalBatch directories
        pattern_files = glob.glob("results/FinalBatch*/patterns/*_patterns.json")
        task_pattern_file = None

        # Find the matching pattern file for the task
        for file in pattern_files:
            if task_name.lower().replace(" ", "_") in file.lower():
                task_pattern_file = file
                break

        if task_pattern_file:
            with open(task_pattern_file, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading patterns: {str(e)}")
        return None


def display_flag(task: str, file_obj, sheet_name=None):
    """Given a task and uploaded file, build insights table, summary, and plots."""
    # Load the dataset
    base_dataset = load_dataset(file_obj, sheet_name)

    if "error" in base_dataset.columns:
        error_msg = base_dataset["error"].iloc[0]
        return (
            f"**Error:** {error_msg}",
            pd.DataFrame(),
            pd.DataFrame(),
            "No summary available due to error loading dataset.",
            "No task information available due to error loading dataset.",
        )

    # --- Metadata as Markdown ---
    md_lines = [
        f"### Dataset Information",
        f"**Selected Task:** {task}",
        f"**Dataset Shape:** {base_dataset.shape}",
        f"**Columns:** {', '.join(base_dataset.columns)}",
    ]
    if sheet_name:
        md_lines.insert(1, f"**Selected Sheet:** {sheet_name}")
    metadata_md = "\n\n".join(md_lines)

    # --- Preview Base Dataset ---
    df_preview = base_dataset.head(10).copy()
    # Reset index to start from 1 for better readability
    df_preview.index = range(1, len(df_preview) + 1)

    # --- Generate basic insights ---
    insights = []

    # Basic statistics for numeric columns
    numeric_cols = base_dataset.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        stats = base_dataset[col].describe()
        insights.append(
            {
                "Data Type": "Numeric",
                "Question": f"What are the basic statistics for {col}?",
                "Actionable Insight": f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}",
            }
        )

    # Basic counts for categorical columns
    categorical_cols = base_dataset.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in categorical_cols:
        value_counts = base_dataset[col].value_counts().head(5)
        insights.append(
            {
                "Data Type": "Categorical",
                "Question": f"What are the top 5 values in {col}?",
                "Actionable Insight": f"Top values: {', '.join([f'{k} ({v})' for k, v in value_counts.items()])}",
            }
        )

    insights_df = pd.DataFrame(insights)

    # --- Summary ---
    summary_md = f"""
    ## Dataset Overview
    - **Number of rows:** {len(base_dataset)}
    - **Number of columns:** {len(base_dataset.columns)}
    - **Numeric columns:** {len(numeric_cols)}
    - **Categorical columns:** {len(categorical_cols)}
    """

    # --- Load and format task patterns ---
    task_info = load_task_patterns(task)
    task_info_md = "No task information available."

    if task_info:
        task_info_lines = ["## Task Information"]

        # KPIs
        if "kpis" in task_info:
            task_info_lines.append("\n### Key Performance Indicators (KPIs)")
            for kpi in task_info["kpis"]:
                task_info_lines.append(f"#### {kpi['name']}")
                task_info_lines.append(f"{kpi['description']}\n")

        # Patterns
        if "patterns" in task_info:
            task_info_lines.append("\n### Patterns")
            for pattern in task_info["patterns"]:
                task_info_lines.append(
                    f"#### {pattern['pattern_index']}: {pattern['pattern']}"
                )
                task_info_lines.append(
                    f"**Columns Involved:** {', '.join(pattern['columns_involved'])}"
                )
                task_info_lines.append(f"**Reasoning:** {pattern['reasoning']}")
                task_info_lines.append(
                    f"**Relevance to KPI:** {pattern['relevance_to_kpi']}\n"
                )

        # Questions and Answers
        if "questions" in task_info and "answers" in task_info:
            task_info_lines.append("\n### Questions and Answers")
            for q, a in zip(task_info["questions"], task_info["answers"]):
                task_info_lines.append(f"#### {q['Question_index']}: {q['question']}")
                task_info_lines.append(f"**KPI:** {q['kpi']}")
                task_info_lines.append(
                    f"**Required Columns:** {', '.join(q['columns_required'])}"
                )
                task_info_lines.append(f"**Algorithm:** {q['algorithm']}")
                task_info_lines.append(f"**Answer:** {a['answer_after_injection']}\n")

        task_info_md = "\n".join(task_info_lines)

    return metadata_md, df_preview, insights_df, summary_md, task_info_md


def on_file_upload(file_obj):
    """Handle file upload and return sheet names if Excel file."""
    if file_obj is None:
        return gr.update(
            choices=[],
            value=None,
            visible=False,
            label="Select Sheet (Excel files only)",
        )

    file_name = file_obj.name.lower()
    if file_name.endswith((".xls", ".xlsx")):
        sheets = get_excel_sheets(file_obj)
        if len(sheets) > 1:
            return gr.update(
                choices=sheets,
                value=sheets[0],
                visible=True,
                label=f"Select Sheet from {file_obj.name}",
            )
    return gr.update(
        choices=[], value=None, visible=False, label="Select Sheet (Excel files only)"
    )


def get_task_skills(task):
    """Get the skills associated with a task."""
    if task in skills_dict:
        return "\n".join([f"• {skill}" for skill in skills_dict[task]])
    return "No skills found for this task."


# Custom CSS for professional styling
custom_css = """
    .gradio-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .gradio-interface {
        padding: 2rem;
        min-height: 100vh;
    }
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        color: #2c3e50 !important;
        text-align: center !important;
    }
    h2 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: #2c3e50 !important;
    }
    h3 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        color: #2c3e50 !important;
    }
    h4 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        color: #2c3e50 !important;
    }
    .gr-form {
        background: #f8f9fa !important;
        padding: 2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin-bottom: 2rem !important;
    }
    .gr-input {
        font-size: 1.1rem !important;
    }
    .gr-button {
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
    }
    .gr-tabs {
        margin-top: 2rem !important;
    }
    .gr-tab-nav {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .gr-box {
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .gr-dataframe {
        font-size: 1rem !important;
    }
    .gr-markdown {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    .gr-file-upload {
        font-size: 1.1rem !important;
    }
    .gr-dropdown {
        font-size: 1.1rem !important;
    }
"""

# --- Build Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
    # BigSnowBench
    
    Welcome to BigSnowBench! Select an analysis task to view its associated skills and algorithms.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=available_tasks,
                label="Select Analysis Task",
                value=available_tasks[0] if available_tasks else None,
                interactive=True,
            )

    with gr.Row():
        with gr.Column():
            skills_output = gr.Markdown(
                label="Associated Skills and Algorithms",
                value="Select a task to view its associated skills and algorithms.",
            )

    with gr.Tabs() as tabs:
        with gr.TabItem("📋 Dataset Information"):
            meta_out = gr.Markdown()
        with gr.TabItem("📊 Data Preview"):
            csv_out = gr.Dataframe(datatype="pandas")
        with gr.TabItem("📈 Basic Statistics"):
            table_out = gr.Dataframe(datatype="pandas")
        with gr.TabItem("📝 Summary"):
            sum_out = gr.Markdown()
        with gr.TabItem("🎯 Task Information"):
            task_info_out = gr.Markdown()

    # connect
    file_input.change(on_file_upload, inputs=[file_input], outputs=[sheet_dropdown])

    task_dropdown.change(
        display_flag,
        inputs=[task_dropdown, file_input, sheet_dropdown],
        outputs=[meta_out, csv_out, table_out, sum_out, task_info_out],
    )

    file_input.change(
        display_flag,
        inputs=[task_dropdown, file_input, sheet_dropdown],
        outputs=[meta_out, csv_out, table_out, sum_out, task_info_out],
    )

    sheet_dropdown.change(
        display_flag,
        inputs=[task_dropdown, file_input, sheet_dropdown],
        outputs=[meta_out, csv_out, table_out, sum_out, task_info_out],
    )

    # Connect the task selection to the skills display
    task_dropdown.change(
        fn=get_task_skills,
        inputs=[task_dropdown],
        outputs=[skills_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
