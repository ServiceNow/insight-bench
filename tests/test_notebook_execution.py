"""
Test using nbformat to execute notebooks in data/notebooks
"""

import nbformat
import json

import os
import glob
from nbconvert.preprocessors import ExecutePreprocessor


def execute_notebooks():
    notebook_dir = "data/notebooks"
    failed_notebooks = {}
    total_notebooks = 0
    successful_notebooks = 0

    # Load previously failed notebooks if the file exists
    if os.path.exists("results/failed_notebooks.json"):
        with open("results/failed_notebooks.json", "r") as f:
            failed_notebooks = json.load(f)
        notebook_files = list(failed_notebooks.keys())
        print("Re-running previously failed notebooks.")
    else:
        # Find all .ipynb files in the notebook directory
        notebook_files = glob.glob(
            os.path.join(notebook_dir, "**", "*.ipynb"), recursive=True
        )
        print("Running all notebooks.")

    for notebook_path in notebook_files:
        total_notebooks += 1
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Set up the execution preprocessor
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

            # Execute the notebook
            ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})

            # Save the executed notebook
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            successful_notebooks += 1
            print("SUCCESS: ", notebook_path)

            # Remove from failed_notebooks if it was there
            # failed_notebooks.pop(notebook_path, None)

        except Exception as e:
            failed_notebooks[notebook_path] = {
                "error": "\n".join(str(e).split("\n")[-3:]),
            }
            print(f"FAILED: {notebook_path}")

        # Update failed_notebooks.json
        with open("results/failed_notebooks.json", "w") as f:
            json.dump(failed_notebooks, f, indent=4)

    success_rate = (
        (successful_notebooks / total_notebooks) * 100 if total_notebooks > 0 else 0
    )

    # Format the failed notebooks string
    failed_notebooks_str = ""
    for notebook, error_info in failed_notebooks.items():
        failed_notebooks_str += f"{notebook}:\n"
        failed_notebooks_str += f"Error: {error_info['error']}\n"
        failed_notebooks_str += "=============\n\n"

    # Save failed_notebooks_str to a text file
    with open("results/failed_notebooks.txt", "w") as f:
        f.write(failed_notebooks_str)

    print(failed_notebooks_str)

    return failed_notebooks_str, success_rate


def test_notebook_execution():
    failed_notebooks, success_rate = execute_notebooks()

    print(f"\nNotebook Execution Results:")
    print(f"Success Rate: {success_rate:.2f}%")

    # Check for failed_notebooks.json
    if os.path.exists("results/failed_notebooks.json"):
        with open("results/failed_notebooks.json", "r") as f:
            failed_notebooks = json.load(f)

        print(f"Failed Notebooks: {len(failed_notebooks)}")
        print(
            f"Total Notebooks: {len(failed_notebooks) + int(success_rate * len(failed_notebooks) / (100 - success_rate))}"
        )

        print("\nFailed Notebooks and Error Messages:")
        for notebook, error_info in failed_notebooks.items():
            print(f"\n{notebook}:")
            print(f"Error: {error_info['error']}")

        assert (
            len(failed_notebooks) == 0
        ), f"Some notebooks failed to execute. Check results/failed_notebooks.json and results/failed_notebooks.txt for details."
    else:
        print("No failed notebooks.")
        assert (
            success_rate == 100
        ), "Expected 100% success rate, but some notebooks may have failed without generating results/failed_notebooks.json"


if __name__ == "__main__":
    test_notebook_execution()
