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

        Args:in
            patterns: The patterns to inject. It is a json file with the following format:
            {
                "kpis": [...],
                "patterns": [
                    {
                        "id": "pattern_id",
                        "pattern": "description of the pattern",
                        "columns_involved": ["col1", "col2", ...],
                        "reasoning": "explanation of why this pattern is useful",
                        "relevance_to_kpi": "how this pattern helps with the task",
                        "benchmark_value": "value to test against"
                    }
                ],
                "QA": [
                    {
                        "question": "question text",
                        "answer_after_injection": "answer text",
                        "caused_by_pattern": "pattern_id"
                    }
                ]
            }

        Returns:
            The code to inject the pattern into the data.
        """

        print("Started getting inject codes ...")

        patterns_dict = json.loads(patterns)
        pattern_info = patterns_dict.get("patterns", [])[0]  # Get the single pattern
        pattern_id = pattern_info.get("pattern_index", "")

        # Find questions specific to this pattern from Answers
        questions = [
            q
            for q in patterns_dict.get("QA", [])
            if q.get("caused_by_pattern") == pattern_id
        ]

        # Format questions and answers
        qa_context = ""
        if questions:
            qa_context = "\nQuestions and Answers for this pattern:\n"
            for q_idx, qa in enumerate(questions, 1):
                qa_context += f"\nQ{q_idx}: {qa.get('question', '')}\n"
                qa_context += f"A{q_idx}: {qa.get('answer_after_injection', '')}\n"

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

        prompt_template = """
        You are a highly skilled data scientist. Your task is to carefully reason through a **pattern injection task** and generate a **valid, executable Python function** that modifies the provided DataFrame according to a given transformation specification.

        You are provided with a pandas DataFrame named `df` with the following columns:

            {columns_str}

        Your objective is to **inject a specific pattern** into the dataset such that:

        - **Pattern**: {pattern_description}
        - **Reasoning**: {reasoning}
        - **Relevance**: {relevance}

        The pattern must be introduced in a **realistic and data-consistent** way.

        You are also given **question–answer pairs** that must hold true **after** the pattern is injected:

        {qa_context}
        ---

       
        ## 🔍 Step-by-Step Reasoning (Think step-by-step before coding)

        1. **Understand the Pattern**
        - Parse the **Pattern** and determine what statistical or structural change is needed.
        - Identify which column(s) in `{columns_str}` should be modified to reflect this change.
        Break down what type of transformation is needed:
        - Is it numeric (e.g., add anomaly, inject trend)?
        - Is it categorical (e.g., modify class labels)?
        - Is it temporal (e.g., adjust timestamps)?
        - Is it group-based, user-based, or global?

        2. **Preserve Answer Validity**
        - Read and interpret each question and expected answer in the QA context.
        - Determine how to inject the pattern so that **answers are correct under the transformed dataset**.

        3. **Plan the Injection Logic**
        - Choose whether to modify, add, or shift rows or values.
        - Ensure the transformation is realistic (e.g., no nulls unless part of the pattern, maintain types).

        4. **Verify Consistency**
        - After applying the transformation:
            - The data should reflect the described pattern.
            - All questions in the QA context must return the expected answers.

        5. **Write Self-contained Code**
        - Use only standard pandas, numpy, or datetime libraries.
        - Do **not** rely on any external variables or undefined data.
        - Your function should be deterministic and runnable independently.

        ## ✅ Output Format

        Generate a complete, **bug-free** Python function with this signature:

        ```python
        def {function_name}(df: pd.DataFrame) -> pd.DataFrame:

        """
        prompt = prompt_template.format(
            columns_str=columns_str,
            pattern_description=pattern_description,
            reasoning=reasoning,
            relevance=relevance,
            qa_context=qa_context,
            function_name=function_name,
        )

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

        output = {
            f"Pattern_{pattern_id}_{'_'.join(columns)}": response.choices[
                0
            ].message.content.strip()
        }
        print(
            f"Finished getting inject codes for pattern {pattern_id} on columns: {'_'.join(columns)}"
        )

        return output, columns_str

    def check_columns_modified(
        self, preinject_data: pd.DataFrame, injected_data: pd.DataFrame, column_str: str
    ) -> dict:
        """Check if the specified columns were modified by comparing pre and post injection statistics.

        Args:
            preinject_data: DataFrame before pattern injection
            injected_data: DataFrame after pattern injection
            column_str: String containing column names to check

        Returns:
            dict: Dictionary mapping column names to their pass/fail status
        """
        # Get the columns to check from column_str
        columns_to_check = []
        for line in column_str.split("\n"):
            if line.strip():
                # Remove any quotes and whitespace
                col = line.strip().strip("'").strip('"').strip()
                if col:  # Only add non-empty strings
                    columns_to_check.append(col)

        # Validate columns exist in both DataFrames
        valid_columns = []
        for col in columns_to_check:
            if col in preinject_data.columns and col in injected_data.columns:
                valid_columns.append(col)
        #     else:
        #         print(f"Warning: Column '{col}' not found in one or both DataFrames")
        #         print(f"Available columns: {list(preinject_data.columns)}")

        if not valid_columns:
            print("Error: No valid columns found to check")
            return {col: "FAIL" for col in columns_to_check}

        # Get stats for specific columns
        pre_stats = preinject_data[valid_columns].describe(include="all")
        post_stats = injected_data[valid_columns].describe(include="all")

        # Compare stats for each column
        results = {}
        for col in columns_to_check:
            if col in columns_to_check:
                # Compare each statistic
                is_modified = False
                # Get available statistics for this column
                available_stats = pre_stats.index.intersection(post_stats.index)
                for stat in available_stats:
                    try:
                        pre_val = pre_stats.loc[stat, col]
                        post_val = post_stats.loc[stat, col]
                        # Handle different types of values
                        if isinstance(pre_val, (int, float)) and isinstance(
                            post_val, (int, float)
                        ):
                            # For numeric values, check if they're different
                            if (
                                abs(pre_val - post_val) > 1e-10
                            ):  # Using small epsilon for float comparison
                                is_modified = True
                                break
                        else:
                            # For non-numeric values, use direct comparison
                            if pre_val != post_val:
                                is_modified = True
                                break
                    except (KeyError, TypeError):
                        # Skip if statistic is not available for this column
                        continue
                results[col] = "PASS" if is_modified else "FAIL"
            else:
                results[col] = "FAIL"

        return results

    def test_inject(
        self,
        columns_str: str,
        injected_data: pd.DataFrame,
        question: str,
        expected_answer: str,
        hash_id: str = None,
        pattern_id: str = None,
        injected_path: str = None,
    ) -> bool:
        """Test if the injected pattern maintains the expected answer to the question.

        Args:
            injected_data: The data after pattern injection
            question: The question to test
            expected_answer: The expected answer to the question
            hash_id: The hash ID for saving test results

        Returns:
            bool: True if the answer matches the expected answer, False otherwise
        """
        print(f"\nTesting question: {question}")
        print(f"Expected answer: {expected_answer}")
        Failed_dataset = []
        preinject_data = pd.read_excel("data/incident.xlsx", sheet_name="incident data")

        print("TEST 1: Are the intended columns modified?")
        test1_results = self.check_columns_modified(
            preinject_data, injected_data, columns_str
        )

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        if hash_id:
            os.makedirs(f"results/{hash_id}", exist_ok=True)

        # Calculate overall test result
        modified_columns = sum(
            1 for status in test1_results.values() if status == "PASS"
        )
        overall_result = "PASS" if modified_columns > 0 else "FAIL"
        print(
            f"\nOverall Test Result: {overall_result} (Modified columns: {modified_columns})"
        )

        # Prepare the test result entry
        test_result = {
            "pattern_id": pattern_id if "pattern_id" in locals() else "unknown",
            "question": question,
            "expected_answer": expected_answer,
            "injection_results": {
                "level_1": {
                    "column_modifications": test1_results,
                    "summary": {
                        "total_columns": len(test1_results),
                        "modified_columns": modified_columns,
                        "unmodified_columns": sum(
                            1 for status in test1_results.values() if status == "FAIL"
                        ),
                    },
                    "overall_result": overall_result,
                }
            },
        }

        # Save or append results to JSON file
        injectoutput_path = (
            f"results/{hash_id}/Test_inject.json"
            if hash_id
            else "results/Test_inject.json"
        )

        # Load existing results if file exists
        if os.path.exists(injectoutput_path):
            with open(injectoutput_path, "r") as f:
                try:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
                except json.JSONDecodeError:
                    existing_results = []
        else:
            existing_results = []

        # Append new result
        existing_results.append(test_result)

        # Save updated results
        with open(injectoutput_path, "w") as f:
            json.dump(existing_results, f, indent=4)

        # Track failed datasets
        if overall_result == "FAIL" and injected_path:
            Failed_dataset.append(injected_path)
            return False

        column_list = list(injected_data.columns)
        summary_stats = injected_data.describe(include="all").to_string()

        print("TEST 2: Are the answers modified?")

        prompt = f"""
        You are a data analysis expert. Your task is to **write a correct and bug-free Python function** that answers a specific question using only the **summary statistics**, **column names**, and **data context** provided for a pandas DataFrame named `df`.
        You will **not** be given the full data, but you will reason about its structure from metadata and statistics.


        ### Provided Information:
        - **Question:**  
        `{question}`

        - **Column Names:**  
        `{column_list}`

        - **Summary Statistics:**  
        `{summary_stats}`

        ### 🔍 Internal Step-by-Step Reasoning (Think carefully before coding)

        1. **Understand the Question**
        - Identify the metric, aggregation, or condition the question is asking about.
        - Determine whether the question refers to counts, proportions, averages, extremes, logical conditions, etc.

        2. **Identify Relevant Columns**
        - Examine which columns in `{column_list}` are directly involved in answering the question.
        - Use summary statistics to infer possible column types (numeric, categorical, datetime).

        3. **Design the Computation Logic**
        - Based on `{summary_stats}`, estimate what operations will lead to the correct answer:
            - Aggregation (e.g., mean, sum, count)
            - Filtering (e.g., rows matching a condition)
            - Grouping or comparison across categories
        - Make conservative assumptions from the statistics — never use columns or values not mentioned.

        4. **Write Clean, Self-Contained Code**
        - Use only pandas and numpy.
        - Do not make assumptions about external data or hardcoded values.
        - Assume that `df` conforms to the described schema and statistics.

        5. **Produce a Valid Return Value**
        - Assign the result to a variable `final_answer` (as a string) and return it.
        - Make sure `final_answer` answers the question **precisely and concisely** (e.g., `"42"`, `"Yes"`, `"25.3%"`).

        ---
       ### ✅ Output Instructions:

        Generate a complete Python function with the following signature:

        ```python
        def answer_question(df: pd.DataFrame) -> str:
        """

        try:
            # Get code to answer the question
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis coding expert. Your task is to write code that answers questions about data. Always respond with valid Python Code.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            raw_code = response.choices[0].message.content.strip()
            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = match.group(1).strip() if match else raw_code.strip()

            code = re.sub(
                r"^\s*import\s+pandas\s+as\s+pd\s*\n?", "", code, flags=re.MULTILINE
            )

            # Create a namespace to execute the code
            namespace = {}

            # Add the full code with main function
            full_code = f"""
import pandas as pd
import sys
from io import StringIO

{code}

def main(df):
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Execute the function
    result = answer_question(df)
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Get the output
    output = mystdout.getvalue().strip()
    
    # If there was printed output, use that, otherwise use the return value
    return output if output else result
"""

            # Retry logic for code execution
            max_code_retries = 3
            code_retry_count = 0
            code_execution_success = False
            actual_answer = None

            while code_retry_count < max_code_retries and not code_execution_success:
                try:
                    if code_retry_count > 0:
                        print(
                            f"\nRetrying code execution (attempt {code_retry_count + 1} of {max_code_retries})"
                        )
                        # Regenerate code with error information
                        error_prompt = f"""
                        Previous code failed with error: {str(last_error)}
                        Please fix the following code to handle the error:
                        {code}
                        """

                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a data analysis coding expert. Your task is to fix the code that failed with the given error. Always respond with valid Python Code.",
                                },
                                {"role": "user", "content": error_prompt},
                            ],
                        )

                        # Update the code with the fixed version
                        fixed_code = response.choices[0].message.content.strip()
                        match = re.search(r"```python(.*?)```", fixed_code, re.DOTALL)
                        if match:
                            fixed_code = match.group(1).strip()

                        # Update full_code with the fixed version
                        full_code = f"""
import pandas as pd
import sys
from io import StringIO

{fixed_code}

def main(df):
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Execute the function
    result = answer_question(df)
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Get the output
    output = mystdout.getvalue().strip()
    
    # If there was printed output, use that, otherwise use the return value
    return output if output else result
"""

                    # Execute the code to define the function
                    exec(full_code, namespace)

                    # Get the main function from the namespace
                    main_function = namespace.get("main")

                    if main_function is None:
                        raise ValueError(
                            "Could not find main function in the generated code"
                        )

                    # Execute the main function on the injected data
                    actual_answer = main_function(injected_data)
                    preinject_answer = main_function(preinject_data)
                    code_execution_success = True

                except Exception as e:
                    last_error = e
                    code_retry_count += 1
                    print(
                        f"Error in code execution (attempt {code_retry_count}/{max_code_retries}): {str(e)}"
                    )

                    if code_retry_count == max_code_retries:
                        print(
                            f"Failed to execute code after {max_code_retries} attempts"
                        )
                        break

            # if not code_execution_success:
            #     return False

            # Create a prompt to check if the answers match
            check_prompt = f"""
            You are a senior data analysis expert. Your task is to evaluate whether two answers to a data analysis question convey the same underlying insight, and whether a pattern injection successfully changed the answer's meaning.

            ---

            ## 🔍 Input Context

            - **Question**:  
            {question}

            - **Expected Answer (Post-injection Ground Truth)**:  
            {expected_answer}

            - **Actual Answer (Produced by the model/code)**:  
            {actual_answer}

            - **Pre-injection Answer (Original before data was modified)**:  
            {preinject_answer}

            - **Code used to produce Actual Answer**:
            ```python
            {code}

            🧠 Step-by-Step Evaluation Criteria
            Step 1: "Expected vs Actual Answer"
            Focus on patterns, trends, and relationships expressed in the answer (e.g., X > Y, positive correlation, delay exists, drop in frequency).

            Consider the answers "Similar" if:

            They reflect the same analytical conclusion or pattern (e.g., "X takes longer than Y")

            Even if specific numbers or stylistic commentary are different or missing

            Consider them "Different" only if:

            The core insight, trend, or relationship has changed

            The actual answer implies a contradictory or unrelated conclusion

            Step 2: "Pre-injection vs Actual Answer"
            These should be Different — i.e., the Actual answer should reflect a change due to the injected pattern.

            If Actual and Pre-injection express the same pattern or conclusion, mark as "Similar" (indicating the injection failed).

            If the pattern or conclusion has shifted (e.g., from X < Y → X > Y), mark as "Different".

            Return your evaluation in strict JSON format inside a json code block. Provide a step-by-step explanation for each judgment. Here is the format:
            {{
            "Expected vs Actual Answer": "Similar" or "Different",
            "Pre-injection vs Actual Answer": "Similar" or "Different",
            "reasoning": "Provide a short step-by-step explanation. Justify both comparisons using values or insights from the answers."
            }}
            ⚠️ Only return the JSON block. Do not include any extra explanation, markdown, or surrounding text.
            """

            # Check if the answers match
            check_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert. Your task is to compare and validate answers to data analysis questions.",
                    },
                    {"role": "user", "content": check_prompt},
                ],
            )

            validation_result = check_response.choices[0].message.content.strip()
            match = re.search(
                r"```json(.*?)```", validation_result, re.DOTALL | re.IGNORECASE
            )
            if not match:
                raise ValueError("No JSON code block found in the response.")

            json_str = match.group(1).strip()

            try:
                validation_data = json.loads(json_str)

                # Update test_result with Test 2 results
                test_result["injection_results"]["level_2"] = {
                    "pre_injection_answer": preinject_answer,
                    "actual_answer": actual_answer,
                    "validation": validation_data,
                    "overall_result": (
                        "PASS"
                        if validation_data["Expected vs Actual Answer"] == "Similar"
                        or validation_data["Pre-injection vs Actual Answer"]
                        == "Different"
                        else "FAIL"
                    ),
                }
                print(f"Test 2 results: {test_result['injection_results']['level_2']}")

                # Save or append results to JSON file
                injectoutput_path = (
                    f"results/{hash_id}/Test_inject.json"
                    if hash_id
                    else "results/Test_inject.json"
                )

                # Load existing results if file exists
                if os.path.exists(injectoutput_path):
                    with open(injectoutput_path, "r") as f:
                        try:
                            existing_results = json.load(f)
                            if not isinstance(existing_results, list):
                                existing_results = [existing_results]
                        except json.JSONDecodeError:
                            existing_results = []
                else:
                    existing_results = []

                # Append new result
                existing_results.append(test_result)

                # Save updated results
                with open(injectoutput_path, "w") as f:
                    json.dump(existing_results, f, indent=4)

                # Return False if Test 2 failed
                if (
                    test_result["injection_results"]["level_2"]["overall_result"]
                    == "FAIL"
                ):
                    return False

                return True

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")

        except Exception as e:
            print(f"Error testing injection: {str(e)}")
            return False

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

            # Retry logic for pattern injection
            max_retries = 3
            retry_count = 0
            last_error = None

            while retry_count < max_retries:
                try:
                    # Always write the script file (in case of previous failures)
                    with open(script_path, "w") as f:
                        f.write(final_code)
                    print(f"Created/Updated script for pattern: {pattern_name}")

                    # Step 4: Run the script
                    subprocess.run(["python3", script_name], check=True, cwd=temp_dir)

                    # Update the DataFrame with the modified data
                    df = pd.read_csv(temp_csv_path)
                    print(f"Successfully injected pattern: {pattern_name}")
                    break

                except Exception as e:
                    last_error = e
                    retry_count += 1
                    print(
                        f"Error injecting pattern (attempt {retry_count}/{max_retries}): {str(e)}"
                    )

                    if retry_count < max_retries:
                        # Regenerate code with error information
                        error_prompt = f"""
                        Previous code failed with error: {str(e)}
                        Please fix the following code to handle the error:
                        {final_code}
                        """

                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a data pattern injection coding expert. Your task is to fix the code that failed with the given error. Always respond with valid Python Code.",
                                },
                                {"role": "user", "content": error_prompt},
                            ],
                        )

                        # Update the code with the fixed version
                        fixed_code = response.choices[0].message.content.strip()
                        match = re.search(r"```python(.*?)```", fixed_code, re.DOTALL)
                        if match:
                            fixed_code = match.group(1).strip()

                        # Update final_code with the fixed version
                        final_code = "import pandas as pd\n" + fixed_code.strip() + "\n"
                        final_code += """if __name__ == "__main__":\n"""
                        final_code += f"""   df = pd.read_csv("temp_data.csv")\n"""
                        final_code += f"""   df = {func_name}(df)\n"""
                        final_code += f"""   df.to_csv("temp_data.csv", index=False)"""

                        print(f"Regenerated code for pattern: {pattern_name}")
                    else:
                        print(
                            f"Failed to inject pattern after {max_retries} attempts. Last error: {str(last_error)}"
                        )
                        print("Continuing with next pattern...")
                        continue

        print("Finished injecting patterns for all patterns.")

        # Step 5: Clean up
        # Don't remove the directory since we want to keep the code files
        # shutil.rmtree(temp_dir)

        return df
