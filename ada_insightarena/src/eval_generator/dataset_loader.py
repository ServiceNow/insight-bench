import pandas as pd
import json
import os
import io
import chardet

class DatasetLoader:
    def __init__(self, csvs_directory, jsons_directory):
        self.csvs_directory = csvs_directory
        self.jsons_directory = jsons_directory
        # self.dataset = self.load_dataset()
        # self.meta = self.load_meta()
        self.dataset=self.load_dataset()
        self.meta=self.load_meta()
    
    # Detect the character encoding of a file
    def detect_encoding(self,file_path):
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read())
            return result['encoding']

    def load_dataset(self):
        # Load the dataset (CSV file)
        for file in os.listdir(self.csvs_directory):
            if file == "data.csv":
                try:
                    encoding =self.detect_encoding(os.path.join(self.csvs_directory, file))
                    return pd.read_csv(os.path.join(self.csvs_directory, file),encoding=encoding)
                except UnicodeDecodeError:
                    # Try a different approach if initial read fails
                    # Handling decode errors by reading as binary and decoding manually
                    with open(os.path.join(self.csvs_directory, file), 'rb') as f:
                        content = f.read()  # Read the file contents as binary
                    # Decode using utf-8 and replace errors
                    decoded_content = content.decode('utf-8', errors='replace')
                    # Use StringIO to convert the decoded content back into a file-like object
                    return pd.read_csv(io.StringIO(decoded_content))
        raise FileNotFoundError("No CSV file found in the directory")

    def load_meta(self):
        # Load the meta file (JSON file)
        for file in os.listdir(self.jsons_directory):
            if file == "meta.json":
                with open(os.path.join(self.jsons_directory, file), 'r') as f:
                    return json.load(f)
        raise FileNotFoundError("No JSON meta file found in the directory")
    
    def load_data(self):
        self.dataset = self.load_dataset()
        self.meta = self.load_meta()
        if self.meta is None:
            directory_id = os.path.basename(self.csvs_directory)  # Assuming directory name as ID
            print(f"Skipping directory due to missing files: {directory_id}")

    def get_summary(self):
        # Generate summary statistics and description
        buffer = io.StringIO()
        self.dataset.info(buf=buffer)
        info = buffer.getvalue()
        
        description = self.meta.get('description', 'No description available')
        summary = self.dataset.describe(include='all').to_string()
        head = self.dataset.head(5).to_string()
        tail = self.dataset.tail(5).to_string()

        return (
            f"Dataset Description:\n{description}\n\n"
            f"Dataset Info:\n{info}\n\n"
            f"Dataset Summary Statistics:\n{summary}\n\n"
            f"Dataset Head (first 5 rows):\n{head}\n\n"
            f"Dataset Tail (last 5 rows):\n{tail}\n"
        )