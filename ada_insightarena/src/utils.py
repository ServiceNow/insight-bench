from openai import OpenAI
import hashlib
import pandas as pd
from pydantic import BaseModel

llm_client = OpenAI()


def get_llm_response(prompt: str):
    # Get response from LLM
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    # Get the generated code and extract code between triple backticks
    full_response = response.choices[0].message.content
    return full_response


def get_llm_response_with_schema(prompt: str, schema: BaseModel):
    # Get response from LLM
    response = llm_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=schema,
    )
    return response


def get_exp_hash(exp_dict):
    # Sort exp_dict by keys, convert to string
    sorted_exp_dict = "_".join(
        [str(k) + str(v) for k, v in sorted(exp_dict.items(), key=lambda item: item[0])]
    )
    # Encode the string before hashing
    hash_obj = hashlib.md5(sorted_exp_dict.encode("utf-8"))
    return hash_obj.hexdigest()

def check_and_fix_dataset(file_path):
    """
    Checks if the dataset is loaded properly or is in a string-like format. 
    Fixes the dataset if needed.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A properly loaded and parsed DataFrame.
    """
    try:
        # Attempt to read the dataset normally
        df = pd.read_csv(file_path)
        # print(df.columns[0], "," in df.columns[0])
        # Check if the DataFrame has only one column with string-like rows
        if df.shape[1] == 1 and "," in df.iloc[0, 0]:
            print(f"Detected malformed dataset: {file_path}")
            
            # Re-read the file properly by parsing lines
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            # Remove outer quotes (single or double) and split by comma
            data = [
                line.strip().strip('"').strip("'").split(",") for line in lines
            ]
            
            # print(data)
            # Create a new DataFrame from the processed data
            df = pd.DataFrame(data[1:], columns=data[0])

        # Clean column names: Remove any quotes or extra whitespace
        df.columns = df.columns.str.strip().str.strip('"').str.strip("'")

        # Ensure all column values are clean: Remove quotes and whitespace
        df = df.map(
            lambda x: x.strip('"').strip("'") if isinstance(x, str) else x
        )

        # Return the fixed DataFrame
        return df

    except Exception as e:
        print(f"Error while processing file {file_path}: {e}")
        return None

def load_prompt(file_name, **kwargs):
    with open(f'prompts/{file_name}', 'r') as file:
        prompt = file.read()
    return prompt.format(**kwargs)