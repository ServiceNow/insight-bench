import gradio as gr
import json
import os


# Function to load metadata from a JSON file
def load_metadata(vis_id=0):
    json_path = f"data/jsons/{vis_id}/meta.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


# Function to format metadata for display
def format_metadata(metadata):
    md = "## 📊 Dataset Information\n"
    md += f"🏷️ **Domain:** {metadata.get('domain', 'N/A')}\n\n"
    md += (
        "📝 **Dataset Description:**\n"
        + metadata.get("dataset_description", "N/A")
        + "\n\n"
    )
    md += "🔗 **Dataset URLs:**\n"
    for url in metadata.get("dataset_urls", []):
        md += f"- {url}\n"
    md += f"\n📓 **Notebook URL:** {metadata.get('notebook_url', 'N/A')}\n"
    return md


# Function to load and parse results
def load_responses(results_file="results/results.json", vis_id="vis_0"):
    if not os.path.exists(results_file):
        return {}, {}

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data for the selected vis_id
    vis_data = results.get(vis_id, {})
    method_1 = vis_data.get("method_1", {}).get("predictions", "No Data Available")
    method_2 = vis_data.get("method_2", {}).get("predictions", "No Data Available")

    return method_1, method_2


# Gradio function to update the display
def update_interface(vis_id):
    # Load metadata and responses
    metadata = load_metadata(vis_id)
    method_1, method_2 = load_responses(vis_id=f"vis_{vis_id}")

    # Format metadata
    metadata_section = format_metadata(metadata)

    return metadata_section, method_1, method_2


# Gradio interface
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# 🔍 Dataset Results Comparison Tool")

        with gr.Row():
            vis_id = gr.Number(
                label="Visualization ID", value=0, interactive=True
            )
            load_button = gr.Button("Load")

        metadata_display = gr.Markdown("## 📊 Metadata will appear here.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original Output")
                method_1_display = gr.Textbox(label="Original Output", lines=15)

            with gr.Column():
                gr.Markdown("### New Output")
                method_2_display = gr.Textbox(label="New Output", lines=15)

        # Link the button to the update function
        load_button.click(
            update_interface,
            inputs=[vis_id],
            outputs=[metadata_display, method_1_display, method_2_display],
        )

    return app


# Run the app
if __name__ == "__main__":
    app = create_app()
    app.launch()
