import nbformat
import sys
from pathlib import Path


def notebook_to_markdown(notebook_path):
    # Convert notebook path to Path object
    nb_path = Path(notebook_path)

    # Check if file exists and is a notebook
    if not nb_path.exists() or nb_path.suffix != ".ipynb":
        print(f"Error: {notebook_path} is not a valid notebook file")
        return

    # Read the notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Create markdown file path with same name
    md_path = nb_path.with_suffix(".md")

    # Extract code cells and write to markdown
    with open(md_path, "w", encoding="utf-8") as f:
        for cell in nb.cells:
            if cell.cell_type == "code":
                f.write("```python\n")
                f.write(cell.source)
                f.write("\n```\n\n")


if __name__ == "__main__":
    notebook_to_markdown("data/skills/notebooks/kernel-pca.ipynb")
