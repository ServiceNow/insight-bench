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
            # Strip serial numbers from column names
            columns = [
                col.split(". ", 1)[-1] if ". " in col else col
                for col in pattern_info.get("columns_involved", [])
            ]
            pattern_description = pattern_info.get("pattern", "")
            reasoning = pattern_info.get("reasoning", "")
            relevance = pattern_info.get("relevance_to_kpi", "")

            function_name = "modify_" + "_".join(columns)
            columns_str = ""
            for column in columns:
                columns_str += f"'{column}\n"

            prompt = f"""
            You are a highly skilled data scientist. Your task is to reason through a pattern injection task and generate a valid, executable Python function to apply the specified transformation.

            You are given a pandas DataFrame named `df` with the following columns:
                        
                {columns_str}

            Your goal is to implement a data transformation that introduces the following pattern:

            - **Pattern**: {pattern_description}
            - **Reasoning**: {reasoning}
            - **Relevance**: {relevance}

            Please follow this **step-by-step reasoning process** internally before generating code:
            1. Identify which of the columns in `{columns_str}` are directly involved in the pattern.
            2. Determine what type of transformation needs to be applied (e.g., numeric shift, conditional flag, group-based anomaly).
            3. Choose appropriate Python libraries and operations to implement the transformation.
            4. Ensure the logic is valid, self-contained, and does not rely on any undefined variables or external dependencies.

            Then, generate a complete **Python function** with the following signature:

            ```python
            def {function_name}(df: pd.DataFrame) -> pd.DataFrame:
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

            output["Pattern" + str(pattern_index + 1) + "_".join(columns)] = (
                response.choices[0].message.content.strip()
            )
            print(
                f"Finished getting inject codes for pattern on columns: {"_".join(columns)}"
            )

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

            func_name = f"modify_" + "_".join(pattern_name.split("_")[1:]).replace(
                " ", "_"
            )

            # Ensure function name is consistent in the code
            code = re.sub(
                r"def\s+[a-zA-Z0-9_]+\(df: pd\.DataFrame\)",
                f"def {func_name}(df: pd.DataFrame)",
                code,
            )

            final_code = "import pandas as pd\n" + code.strip() + "\n"
            final_code += """if __name__ == "__main__":\n"""
            final_code += f"""   df = pd.read_csv("temp_data.csv")\n"""
            final_code += f"""   df = {func_name}(df)\n"""
            final_code += f"""   df.to_csv("temp_data.csv", index=False)"""

            # Create script in codefiles directory
            script_name = f"{func_name}.py"
            script_path = os.path.join(temp_dir, script_name)

            if os.path.exists(script_path):
                print(
                    f"Skipping script creation for pattern: {pattern_name} - file already exists"  # For demo:Amrutha
                )
            else:
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
