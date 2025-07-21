import os
import json


def generate_json_files(base_dir):
    # Walk through all subfolders and files in the base directory
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Check if the file has .csv or .xlsx extension
            if file.endswith(".csv") or file.endswith(".xlsx"):
                # Extract the file path and name (excluding extension)
                file_path = os.path.join(root, file)
                dataset_name = os.path.splitext(file)[0]

                # Generate questions based on the folder name (task name)
                task_name = os.path.basename(root)
                questions = [
                    f"Forecast future {task_name} trends.",
                    f"Predict seasonal patterns in {task_name}.",
                    f"Detect anomalies in {task_name} data.",
                ]

                # PROMPT: Create a JSON file with the following structure:
                # {
                #     "path": "<relative path to the dataset>",
                #     "questions": ["<list of questions specific to time series models>"]
                # }
                # The JSON file should have the same name as the dataset (excluding file extension) and be saved in the same directory.

                # Create the JSON content
                json_content = {"path": file_path, "questions": questions}

                # Define the output JSON file path
                json_file_path = os.path.join(root, f"{dataset_name}.json")

                # Write the JSON content to the file
                with open(json_file_path, "w") as json_file:
                    json.dump(json_content, json_file, indent=4)

                print(f"Generated JSON file: {json_file_path}")


# Base directory containing the data/ folder
base_directory = "data/"
generate_json_files(base_directory)
