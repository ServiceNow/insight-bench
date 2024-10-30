import os
import pandas as pd
from insightbench.utils.agent_utils import analysis_nb_to_gt


def get_benchmark(dataset_type, datadir):
    """load a set of tables and a set of golden insights to evaluate the agent's performance"""

    if dataset_type == "toy":
        flag_list = list(range(1, 6))
    elif dataset_type == "standard":
        flag_list = list(range(1, 31))
    elif dataset_type == "full":
        flag_list = list(range(1, 101))
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")

    # Create a list of dictionaries with notebook and csv paths
    notebooks_list = [
        {
            "notebook_path": os.path.join(datadir, f"flag-{flag}.ipynb"),
            "dataset_csv_path": os.path.join(datadir, "csvs", f"flag-{flag}.csv"),
            # Check if the sysuser CSV exists and add it to the dictionary if it does
            "user_dataset_csv_path": (
                os.path.join(datadir, "csvs", f"flag-{flag}-sysuser.csv")
                if os.path.exists(
                    os.path.join(datadir, "csvs", f"flag-{flag}-sysuser.csv")
                )
                else None
            ),
        }
        for flag in flag_list
    ]

    return notebooks_list


def load_dataset_dict(
    dataset_csv_path, dataset_notebook_path, user_dataset_csv_path=None
):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_csv_path)

    # Use the analysis_nb_to_gt function to extract insights from the notebook
    notebook_data = analysis_nb_to_gt(dataset_notebook_path)

    # Create a dictionary to store the dataset
    dataset_dict = {
        "dataset_csv_path": dataset_csv_path,
        "user_dataset_csv_path": user_dataset_csv_path,
        "dataframe": df,
        "notebook": notebook_data,
        "insights": [
            ins["insight_dict"]["insight"] for ins in notebook_data["insights"]
        ],
        "summary": notebook_data["flag"],
    }

    return dataset_dict
