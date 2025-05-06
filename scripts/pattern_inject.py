from openai import OpenAI
import json
import os
import uuid
import shutil
import re
import subprocess

class PatternInjector:
    def __init__(self, api_key: str = None):
        """Initialize the PatternInjector with OpenAI API key.

        Args:
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        self.client = OpenAI(api_key=api_key)

    def get_inject_codes(self, patterns: str) -> dict:
        """Get the code to inject the pattern into the data.

        Args:
            patterns: The patterns to inject. It is a json file with the following format:
            {
                "column_name_1": [
                    {
                        "pattern": "description of the pattern",
                        "reasoning": "explanation of why this pattern is useful",
                        "relevance_to_task": "how this pattern helps with the task"
                    },
                    ...
                ],
                ...
            }

        Returns:
            The code to inject the pattern into the data.
        """

        print("Started getting inject codes ...")

        patterns = json.loads(patterns)

        output = {}

        for column, pattern in patterns.items():

            prompt = f"""
            You are given a pandas DataFrame named `df` that contains a column called `{column}`.

            Your task is to write a Python function that analyzes the column based on specific patterns and data quality or relevance concerns.

            The function should:
                - Be named `modify_{column}`
                - Be a standalone function
                - Take `df` as its only argument
                - Implement logic that addresses all of the following patterns (described below)
                - Use only standard libraries or common ones such as `pandas`, `numpy`, `re`, etc.
                - Not use any complex or uncommon third-party libraries

            This is the signature of the function:
            ```python
            def modify_{column}(df: pd.DataFrame) -> pd.DataFrame:
            ```

            Here are the patterns you must handle:
            {pattern}

            Output requirements:
                - Only include the necessary `import` statements and the function definition.
                - Do not include any explanation or comments.
                - Only generate one function with all logic embedded.
                - Do not include usage examples or extra text.
                - Function should have a return statement that returns the modified DataFrame.

            This is an example of the expected output for a column called `age`:
            ```python
            import numpy as np
            # And importing any other necessary libraries

            def modify_age(df: pd.DataFrame) -> pd.DataFrame:
                # Example logic to handle the patterns
                df['age'] = df['age'].apply(lambda x: np.nan if x < 0 else x)
                df['age'] = df['age'].fillna(df['age'].mean())
                return df
            ```

            IMPORTANT NOTES:
                - The function should be valid Python code and should not include any comments or explanations.
                - The function should be self-contained and not rely on any external context or variables.
                - The function should be able to handle the patterns described above and return a modified DataFrame.
                - There should be no additional code except for the function definition and necessary imports. You should not write and include other functions or call this functions (not even writing a main function).
                - The function should not include any print statements or logging.

            Please return only the code (imports + one function) in python environment. Nothing else.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o (OpenAI Omni)
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data pattern injection coding expert. Your task is to write correct and simple codes to inject the given patterns to help accomplish specific analytics tasks. Always respond with valid Python Code.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            output[column] = response.choices[0].text.strip()

            print(f"Finished getting inject codes for column: {column}")

        print("Finished getting inject codes for all columns.")

        return output
    
    def inject_patterns(self, data_file_addr:str, pattern_codes: dict):
        """Inject the patterns into the data.

        Args:
            pattern_codes: The pattern codes to inject. It is a dictionary with the following format:
            {
                "column_name_1": "code to inject the pattern",
                ...
            }
            data_file_addr: Address to the original CSV data file.

        Returns:
            Nothing. The function creates a temp folder with modified CSV and scripts.
        """

        print("Started injecting patterns ...")

        # Step 1: Create temp directory
        temp_dir = f"temp_{uuid.uuid4().hex}"
        os.makedirs(temp_dir, exist_ok=True)

        # Step 2: Copy the original CSV
        filename = os.path.basename(data_file_addr)
        temp_csv_path = os.path.join(temp_dir, filename)
        shutil.copy2(data_file_addr, temp_csv_path)

        # Step 3: Create Python scripts for each column
        for column, raw_code in pattern_codes.items():
            print(f"Injecting pattern for column: {column}")

            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = match.group(1).strip() if match else raw_code.strip()

            code = re.sub(r"^\s*import\s+pandas\s+as\s+pd\s*\n?", "", code, flags=re.MULTILINE)

            func_name = f"modify_{column}"

            final_code = "import pandas as pd\n" + code.strip() + "\n"
            final_code += """if __name__ == "__main__":\n"""
            final_code += f"""   df = pd.read_csv("{filename}")\n"""
            final_code += f"""   df = {func_name}(df)")\n"""
            final_code += f"""   df.to_csv("{filename}", index=False)"""

            script_path = os.path.join(temp_dir, f"{func_name}.py")
            with open(script_path, "w") as f:
                f.write(final_code)

            print(f"Created script for column: {column} at {script_path}")

            # Step 4: Run the script
            subprocess.run(["python", script_path], check=True, cwd=temp_dir)

            print(f"Injected pattern for column: {column}")
            
        print("Finished injecting patterns for all columns.")
        
        # Step 5: Copy modified CSV to original directory
        original_dir = os.path.dirname(data_file_addr)
        injected_filename = os.path.splitext(filename)[0] + "_injected.csv"
        injected_path = os.path.join(original_dir, injected_filename)
        shutil.copy2(temp_csv_path, injected_path)

        # Step 6: Clean up
        shutil.rmtree(temp_dir)

        print(f"Injected CSV saved to: {injected_path}")