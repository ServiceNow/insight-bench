import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# 1) find all flag-*.json files under the notebooks folder
JSON_DIR = os.path.join("data", "notebooks")
flag_files = sorted([
    fn for fn in os.listdir(JSON_DIR)
    if fn.startswith("flag-") and fn.endswith(".json")
])

# 2) preload them into a dict
flags = {}
for fn in flag_files:
    path = os.path.join(JSON_DIR, fn)
    with open(path, "r") as f:
        flags[fn] = json.load(f)

def display_flag(fn: str):
    """Given a filename, extract metadata, preview CSV, build insights table, summary, and plots."""
    data = flags[fn]

    # --- Metadata as Markdown ---
    md = data.get("metadata", {})
    md_lines = [
        f"**Goal:** {md.get('goal','')}",
        f"**Role:** {md.get('role','')}",
        f"**Category:** {md.get('category','')}",
        f"**Description:** {md.get('dataset_description','')}",
        f"**Header:** {md.get('header','')}",
    ]
    metadata_md = "\n\n".join(md_lines)

    # --- Summary ---
    summary_md = data.get("summary", "")

    # --- Load & preview CSV ---
    csv_path = data.get("dataset_csv_path", "")
    try:
        df = pd.read_csv(csv_path)
        df_preview = df.head(10)
    except Exception as e:
        df_preview = pd.DataFrame({"error": [f"Could not load CSV: {e}"]})

    # --- Build insights table ---
    insight_list = data.get("insight_list", [])
    table_rows = []
    for ins in insight_list:
        table_rows.append({
            "Data Type":         ins.get("data_type", ""),
            # "Insight":           ins.get("insight", ""),
            "Question":          ins.get("question", ""),
            "Actionable Insight":ins.get("actionable_insight", "")
        })
    insights_df = pd.DataFrame(table_rows)

    # --- Generate any histogram plots ---
    figs = []
    for ins in insight_list:
        plot = ins.get("plot", {})
        if plot.get("plot_type") == "histogram":
            x = plot.get("x_axis", {}).get("value", [])
            y = plot.get("y_axis", {}).get("value", [])
            title = plot.get("title", "")
            xlabel = plot["x_axis"].get("name", "")
            ylabel = plot["y_axis"].get("name", "")

            fig, ax = plt.subplots()
            ax.bar(x, y)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            figs.append(fig)

    # if you have multiple histograms, you can return them all, but Gradio will show only the first by default
    first_fig = figs[0] if figs else None

    return metadata_md, df_preview, insights_df, summary_md, first_fig

# --- Build Gradio UI ---
with gr.Blocks(css="""
    .gradio-container { max-width: 1200px; margin: auto }
""") as demo:
    gr.Markdown("# 🚩 Flag‐by‐Flag Insights Explorer")

    with gr.Row():
        flag_dropdown = gr.Dropdown(
            choices=flag_files,
            label="Select a flag-JSON to view",
            value=flag_files[0] if flag_files else None,
            interactive=True,
        )

    with gr.Tabs():
        with gr.TabItem("Metadata"):
            meta_out = gr.Markdown()
        with gr.TabItem("CSV Preview"):
            csv_out = gr.Dataframe(datatype="pandas")
        with gr.TabItem("Insights Table"):
            table_out = gr.Dataframe(datatype="pandas")
        with gr.TabItem("Summary"):
            sum_out = gr.Markdown()
        with gr.TabItem("Histogram"):
            plot_out = gr.Plot()

    # connect
    flag_dropdown.change(
        display_flag,
        inputs=[flag_dropdown],
        outputs=[meta_out, csv_out, table_out, sum_out, plot_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
