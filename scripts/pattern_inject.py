from openai import OpenAI
import json
import os
import shutil
import re
import subprocess
import pandas as pd


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
                "kpis": [...],
                "patterns": [
                    {
                        "pattern": "description of the pattern",
                        "columns_involved": ["col1", "col2", ...],
                        "reasoning": "explanation of why this pattern is useful",
                        "relevance_to_kpi": "how this pattern helps with the task",
                        "benchmark_value": "value to test against"
                    },
                    ...
                ]
            }

        Returns:
            The code to inject the pattern into the data.
        """

        print("Started getting inject codes ...")

        patterns_dict = json.loads(patterns)
        patterns_list = patterns_dict.get("patterns", [])
        output = {}

        for pattern_index, pattern_info in enumerate(patterns_list):
            columns = pattern_info.get("columns_involved", [])
            pattern_description = pattern_info.get("pattern", "")
            reasoning = pattern_info.get("reasoning", "")
            relevance = pattern_info.get("relevance_to_kpi", "")

            function_name = "modify_" + "_".join(columns)
            columns_str = ""
            for i, column in enumerate(columns):
                columns_str += f"'{i+1}. {column}\n"

            prompt = f"""
            You are given a pandas DataFrame named `df` that contains the following columns: 
            
            `{columns_str}`

            Your task is to write a Python function that modifies the columns based on specific patterns.

            The function should:
                - Be named `{function_name}`
                - Be a standalone function
                - Take `df` as its only argument
                - Implement logic that addresses the following pattern:
                    Pattern: {pattern_description}
                    Reasoning: {reasoning}
                    Relevance: {relevance}
                - Use only standard libraries or common ones such as `numpy`, `re`, etc.
                - Not use any complex or uncommon third-party libraries

            This is the signature of the function:
            ```python
            def {function_name}(df: pd.DataFrame) -> pd.DataFrame:
            ```

            Output requirements:
                - Only include the necessary `import` statements and the function definition.
                - Do not include any explanation or comments.
                - Only generate one function with all logic embedded.
                - Do not include usage examples or extra text.
                - Function should have a return statement that returns the modified DataFrame.

            This is an example of the expected output for columns called `age` and `height`:

            ```python
            import numpy as np
            # And importing any other necessary libraries

            def modify_age_height(df: pd.DataFrame) -> pd.DataFrame:
                # Example logic to handle the patterns
                df['age'] = df['age'].apply(lambda x: np.nan if x < 0 else x)
                df['height'] = df['height'].fillna(df['height'].mean())
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
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data pattern injection coding expert. Your task is to write correct and simple codes to inject the given patterns to help accomplish specific analytics tasks. Always respond with valid Python Code.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            output["Pattern" + str(pattern_index+1) + "_".join(columns)] = response.choices[0].message.content.strip()
            print(f"Finished getting inject codes for pattern on columns: {"_".join(columns)}")

        print("Finished getting inject codes for all patterns.")
        return output

    def inject_patterns(
        self,
        base_df: pd.DataFrame,
        pattern_codes: dict,
        hash_id: str = None,
    ) -> pd.DataFrame:
        """Inject the patterns into the data.

        Args:
            base_df: The base DataFrame to inject the patterns into.
            pattern_codes: The pattern codes to inject. It is a dictionary with the following format:
            {
                "name1": "code to inject for pattern1",
                "name1": "code to inject for pattern2",
                ...
            }
            hash_id: The hash ID to use for the temp directory. If not provided, will use "default".
        """

        print("Started injecting patterns ...")

        # Step 1: Create temp directory inside results/{hash_id}/codefiles/
        if hash_id is None:
            hash_id = "default"
        temp_dir = os.path.join("results", hash_id, "codefiles")
        os.makedirs(temp_dir, exist_ok=True)

        # Step 2: Handle input data
        if isinstance(base_df, pd.DataFrame):
            df = base_df.copy()
            filename = "temp_data.csv"
            temp_csv_path = os.path.join(temp_dir, filename)
            df.to_csv(temp_csv_path, index=False)
        else:
            raise ValueError(
                "base_df should be a pandas DataFrame. Please provide a valid DataFrame."
            )

        # Step 3: Create Python scripts for each column
        for pattern_name, raw_code in pattern_codes.items():
            print(f"Injecting pattern: {pattern_name}")

            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = match.group(1).strip() if match else raw_code.strip()

            code = re.sub(
                r"^\s*import\s+pandas\s+as\s+pd\s*\n?", "", code, flags=re.MULTILINE
            )

            func_name = f"modify_" + "_".join(pattern_name.split("_")[1:])

            final_code = "import pandas as pd\n" + code.strip() + "\n"
            final_code += """if __name__ == "__main__":\n"""
            final_code += f"""   df = pd.read_csv("{filename}")\n"""
            final_code += f"""   df = {func_name}(df)\n"""
            final_code += f"""   df.to_csv("{filename}", index=False)"""

            # Create script in codefiles directory
            script_name = f"{func_name}.py"
            script_path = os.path.join(temp_dir, script_name)

            with open(script_path, "w") as f:
                f.write(final_code)

            print(f"Created script for pattern: {pattern_name}")

            # Step 4: Run the script
            subprocess.run(["python3", script_name], check=True, cwd=temp_dir)

            # Update the DataFrame with the modified data
            df = pd.read_csv(temp_csv_path)

            print(f"Injected pattern for pattern: {pattern_name}")

        print("Finished injecting patterns for all patterns.")

        # Step 5: Clean up
        # Don't remove the directory since we want to keep the code files
        # shutil.rmtree(temp_dir)

        return df
