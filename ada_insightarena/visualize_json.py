import os, json
from dotenv import load_dotenv
from pprint import pprint
import pandas as pd
import exp_configs

# Load environment variables at the start
load_dotenv()


def parse_comparison_text(text):
    """Parse the comparison text into sections."""
    sections = {}
    current_section = None
    current_content = []

    for line in text.split("\n"):
        # Skip separator lines
        if "=" * 10 in line:
            continue

        # Check for section headers
        if line.strip() and all(c == "-" for c in line.strip()):
            continue

        if line.startswith("❓ QUESTIONS"):
            current_section = "questions"
            current_content = []
        elif line.startswith("📌 METADATA"):
            current_section = "metadata"
            current_content = []
        elif line.startswith("📊 PREDICTIONS"):
            current_section = "predictions"
            current_content = []
        elif line.startswith("📋 GROUND TRUTH"):
            current_section = "ground_truth"
            current_content = []
        elif line.strip() and current_section:
            if current_section == "questions":
                # Extract just the question from the dictionary-like string
                if "'question':" in line:
                    question = (
                        line.split("'question': ")[1].split("'answer':")[0].strip("', ")
                    )
                    # Add Roman numeral numbering (i, ii, iii, etc.)
                    question_num = len(current_content) + 1
                    roman_num = f"{question_num})"
                    current_content.append(f"{roman_num} {question}\n\n")
            elif current_section == "predictions":
                current_content.append(line.strip()+"\n\n")
            else:
                current_content.append(line.strip())
            sections[current_section] = "\n".join(current_content)

    return sections


def get_experiment_results(exp_groups):
    """Collect results from all experiments in one or more groups.

    Args:
        exp_groups: str or list of str, experiment group name(s)
    """
    result_list_question_id = []
    result_list_vis_id=[]
    result_list_skill_id=[]

    # Convert single string to list for consistent handling
    if isinstance(exp_groups, str):
        exp_groups = [exp_groups]

    print(f"\nProcessing experiment groups: {exp_groups}")

    n_count = 0 
    # this is for vis particularly
    # Process each experiment group
    for exp_group_name in exp_groups:
        results_dir = f"results/{exp_group_name}"
        print(f"\nLooking for results in: {results_dir}")

        # Check if the results directory exists
        if not os.path.exists(results_dir):
            print(f"No results found for experiment group: {exp_group_name}")
            continue

        # Iterate through all experiment hashes
        exp_hashes = os.listdir(results_dir)
        print(f"Found {len(exp_hashes)} experiment(s) in {exp_group_name}")

        for exp_hash in exp_hashes:

            exp_dir = os.path.join(results_dir, exp_hash)
            if not os.path.isdir(exp_dir):
                continue

            vis_count = 0
            question_count= 0 #Counter for questions in each dataset
            # Find all vis_* directories
            for item in os.listdir(exp_dir):
                if item.isdigit():
                    item_path=os.path.join(exp_dir,item)
                    if os.path.isdir(item_path):
                        for it in os.listdir(item_path):
                            if it.startswith("question_"):  
                                question_count += 1
                                question_id = it.split("_")[1]  # Extract question number (0,1,...)

                                # Path to insights text file inside question folder
                                insight_file = os.path.join(exp_dir, item,it, "question_insight.txt")
                                plot_path = os.path.join(exp_dir, item,it, "plot.jpeg")  # Path to question's plot

                                if os.path.exists(insight_file):
                                    with open(insight_file, "r") as f:
                                        lines = f.readlines()

                                # Extract Question and Answer from the text file
                                question = lines[0].strip().replace("Question: ", "") if len(lines) > 0 else ""
                                # answer = lines[1].strip().replace("Insight: ", "") if len(lines) > 1 else ""
                                answer_start_index = next((i for i, line in enumerate(lines) if line.startswith("Insight:")), None)
                                if answer_start_index is not None:
                                    answer = "".join(lines[answer_start_index:]).replace("Insight: ", "").strip()
                                else:
                                    answer = ""

                                # Store results
                                result_dict = {
                            "exp_group_name": exp_group_name, #insights_w_skills, insights_wo_skills etc.
                            "hash": exp_hash,  # etc. 070e74f1b6f67067a5df97f69dd7526e
                            "dataset_id":item, # Dataset ID (10, 11, etc.)
                            "question_id": question_id,  # Question number (0,1,...)
                            "question": question,  # Extracted Question
                            "insight": answer,  # Extracted Answer (Insight)
                            "plot_path": plot_path  # Path to plot image
                            }
                                result_list_question_id.append(result_dict)
                            else:
                                print(f"No question_insight.txt found for {it} in {exp_dir}!")
                        if "ques_ans.json" in os.listdir(item_path):
                            # if it.startswith("ques_an_"):
                                # question_id=it.split("_")[1]
                            ques_ans_file=os.path.join(exp_dir, item, "ques_ans.json")
                            with open(ques_ans_file, "r", encoding="utf-8") as f:
                                print("Reading from:", ques_ans_file)
                                data_list = json.load(f)
                                # skills_list = data.get("skills", [])
                                # first_skill = skills_list[0] if skills_list else "no_skill"
                                # questions_array = data.get("questions", [])
                            questions_dict = {}

                            for idx,object in enumerate(data_list):
                                q_text = object.get("question", "No question text")
                                skill_field = object.get("skill", [])
                                # if skill_field is a list like ["Skill_A", "Skill_B"], take the first
                                if isinstance(skill_field, list) and skill_field:
                                    first_skill = skill_field[0]
                                else:
                                    first_skill = "no_skill"
                                    
                                # Store under string index
                                questions_dict[str(idx)] = {
                                    "question": q_text,
                                    "skill": first_skill
                                }

                                # for idx, item in enumerate(questions_array):
                                #     q_text = item.get("question", "No question text")
                                #     questions_dict[str(idx)] = {
                                #         "question": q_text,
                                #         "skill": first_skill
                                #     }
                                result_dict = {
                            "exp_group_name": exp_group_name, #insights_w_skills, insights_wo_skills etc.
                            "hash": exp_hash,  # etc. 070e74f1b6f67067a5df97f69dd7526e
                            "dataset_id":item, # Dataset ID (10, 11, etc.)
                            "question_id": idx,  # Question number (0,1,...)
                            "question": questions_dict[str(idx)]["question"],  # Extracted Question
                            "skill": questions_dict[str(idx)]["skill"]  # Extracted Answer (Insight)
                            }
                                result_list_skill_id.append(result_dict)

                                # output_data = {
                                #     f"dataset_{item}": {
                                #         "method_1": {
                                #             "exp_group": method1_exp_group,
                                #             "hash": method1_hash,
                                #             "questions": questions_dict
                                #         },
                                #         "method_2": {
                                #             "exp_group": method2_exp_group,
                                #             "hash": method2_hash,
                                #             "questions": questions_dict
                                #         }
                                #     }
                                # }
                        
                           
                            

                elif item.startswith("vis_"):
                    n_count += 1
                    vis_count += 1
                    vis_i = item.split("_")[1]
                    comparison_file = os.path.join(
                        exp_dir, item, "insights_comparison.txt"
                    )
                    print(f"Processing {comparison_file}")
                    if os.path.exists(comparison_file):
                        with open(comparison_file, "r") as f:
                            comparison_text = f.read()

                        # Parse the comparison text into sections
                        parsed_sections = parse_comparison_text(comparison_text)

                        result_dict = {
                            "exp_group_name": exp_group_name,
                            "hash": exp_hash,
                            "vis_id": vis_i,
                            **parsed_sections,  # Add all sections as separate keys
                        }

                        result_list_vis_id.append(result_dict)
                    else:
                        print(f"No insights_comparison.txt found for {exp_dir}")

            print(f"Experiment {exp_hash}: found {vis_count} visualization(s)")
            print(f"Dataset {exp_hash}: found {question_count} questions!")

    # print(f"\nTotal results collected: {len(result_list)}")
    df1 = pd.DataFrame(result_list_question_id)
    # print(df1.columns)
    df1 = df1.sort_values(by=["dataset_id", "question_id"])

    df2 = pd.DataFrame(result_list_vis_id)
    df2 = df2.sort_values(by="vis_id")

    df3=pd.DataFrame(result_list_skill_id)
    print(df3.head,df3.columns)
    df3=df3.sort_values(by=["dataset_id", "question_id"])


    return result_list_question_id,result_list_vis_id,result_list_skill_id


import os
import json

def save_formatted_results_question(results, output_file="results/results_by_question.json"):
    """
    Saves results in a formatted JSON file organized by:
      dataset_id -> method_X -> { exp_group, hash, questions }
    
    Where each method_X (method_1, method_2, etc.) contains:
      exp_group: name of the method (e.g., insights_w_skills_recent)
      hash: unique hash for that method
      questions: dictionary of question_id -> { question, insight, plot_path }
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Main dictionary to hold everything
    dataset_groups = {}

    for result in results:
        # Extract fields from the result
        dataset_id = f"dataset_{result['dataset_id']}"
        exp_group = result["exp_group_name"]      # e.g. "insights_w_skills_recent"
        question_id = str(result["question_id"])  # Convert to string for JSON keys

        # If we haven't seen this dataset_id yet, initialize it
        if dataset_id not in dataset_groups:
            # _method_map will track which method_X (method_1, method_2, etc.)
            # corresponds to each exp_group name
            dataset_groups[dataset_id] = {
                "_method_map": {},
                "_method_count": 0
            }

        # Check if this exp_group is already mapped to a method_X
        if exp_group not in dataset_groups[dataset_id]["_method_map"]:
            dataset_groups[dataset_id]["_method_count"] += 1
            method_key = f"method_{dataset_groups[dataset_id]['_method_count']}"
            
            # Create a new mapping from exp_group to method_X
            dataset_groups[dataset_id]["_method_map"][exp_group] = method_key
            
            # Initialize the method_X entry
            dataset_groups[dataset_id][method_key] = {
                "exp_group": exp_group,
                "hash": result["hash"],
                "questions": {}
            }
        else:
            # If we already have a mapping, just get the existing method_key
            method_key = dataset_groups[dataset_id]["_method_map"][exp_group]

        # Insert or update the question entry under the correct method
        dataset_groups[dataset_id][method_key]["questions"][question_id] = {
            "question": result["question"],
            "insight": result["insight"],
            "plot_path": result["plot_path"]
        }

    # Clean up the helper fields _method_map and _method_count before writing to JSON
    for ds_id in list(dataset_groups.keys()):
        dataset_groups[ds_id].pop("_method_map", None)
        dataset_groups[ds_id].pop("_method_count", None)

    # Write out the final JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_groups, f, indent=4)

    print(f"Results saved successfully to {output_file}")


def save_formatted_results_skills(results, output_file="results/results_by_skill.json"):
    """
    Saves results in a formatted JSON file organized by:
      dataset_id -> method_X -> { exp_group, hash, questions }
    
    Where each method_X (method_1, method_2, etc.) contains:
      exp_group: name of the method (e.g., insights_w_skills_recent)
      hash: unique hash for that method
      questions: dictionary of question_id -> { question, insight, plot_path }
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Main dictionary to hold everything
    dataset_groups = {}

    for result in results:
        # Extract fields from the result
        dataset_id = f"dataset_{result['dataset_id']}"
        exp_group = result["exp_group_name"]      # e.g. "insights_w_skills_recent"
        question_id = str(result["question_id"])  # Convert to string for JSON keys

        # If we haven't seen this dataset_id yet, initialize it
        if dataset_id not in dataset_groups:
            # _method_map will track which method_X (method_1, method_2, etc.)
            # corresponds to each exp_group name
            dataset_groups[dataset_id] = {
                "_method_map": {},
                "_method_count": 0
            }

        # Check if this exp_group is already mapped to a method_X
        if exp_group not in dataset_groups[dataset_id]["_method_map"]:
            dataset_groups[dataset_id]["_method_count"] += 1
            method_key = f"method_{dataset_groups[dataset_id]['_method_count']}"
            
            # Create a new mapping from exp_group to method_X
            dataset_groups[dataset_id]["_method_map"][exp_group] = method_key
            
            # Initialize the method_X entry
            dataset_groups[dataset_id][method_key] = {
                "exp_group": exp_group,
                "hash": result["hash"],
                "questions": {}
            }
        else:
            # If we already have a mapping, just get the existing method_key
            method_key = dataset_groups[dataset_id]["_method_map"][exp_group]

        # Insert or update the question entry under the correct method
        dataset_groups[dataset_id][method_key]["questions"][question_id] = {
            "question": result["question"],
            "skill": result["skill"]
        }

    # Clean up the helper fields _method_map and _method_count before writing to JSON
    for ds_id in list(dataset_groups.keys()):
        dataset_groups[ds_id].pop("_method_map", None)
        dataset_groups[ds_id].pop("_method_count", None)

    # Write out the final JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_groups, f, indent=4)

    print(f"Results saved successfully to {output_file}")



def save_formatted_results_vis(results, output_file="results/results.json"):
    """Save results in a formatted text file organized by vis_id.

    Args:
        results: List of result dictionaries
        output_file: Path to output file
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Group results by vis_id
    vis_groups = {}
    for result in results:
        vis_id = result["vis_id"]
        if vis_id not in vis_groups:
            vis_groups[vis_id] = []
        vis_groups[vis_id].append(result)

    def wrap_text(text, width=60):
        """Helper function to wrap text at specified width."""
        # Split text into paragraphs first
        paragraphs = text.split('\n\n')
        wrapped_paragraphs = []
        
        for paragraph in paragraphs:
            words = paragraph.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(" ".join(current_line))
            wrapped_paragraphs.append("\n".join(lines))
        
        # Join paragraphs with double newlines
        return "\n\n".join(wrapped_paragraphs)

    results_json = {}
    # Process each vis_id group
    for vis_id in sorted(vis_groups.keys()):
        # f.write(f"vis_{vis_id}\n")
        # f.write("====\n\n")
        temp_dict = {}
        # Write metadata and questions once per vis_id (taking from first result)
        first_result = vis_groups[vis_id][0]
        if "metadata" in first_result:
            # f.write("METADATA:\n--------\n")
            metadata = wrap_text(first_result["metadata"].strip()) 
        if "questions" in first_result:
            
            questions = wrap_text(first_result["questions"].strip())

        # Start building the dictionary
        temp_dict[f"vis_{vis_id}"] = {
            "metadata": metadata,
            "questions": questions,
            "method_1": {},
            "method_2": {},
        }
        # Write each method's results
        for i, result in enumerate(vis_groups[vis_id], 1):
            method_key = f"method_{i}"
            exp_group = result['exp_group_name']
            exp_hash = result['hash']
            wrapped_predictions = wrap_text(result["predictions"].strip())
            temp_dict[f"vis_{vis_id}"][method_key] = {
            "exp_group": exp_group,
            "exp_hash": exp_hash,
            "predictions": wrapped_predictions,
            }
        results_json[f"vis_{vis_id}"] = temp_dict[f"vis_{vis_id}"]
    
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=4)



if __name__ == "__main__":
    exp_groups = ["insights_w_skills_pilot2ndrun","insights_wo_skills_pilot2ndrun"] #, "insights_wo_skills"

    # Collect and display results
    results_question,results_vis,results_skill = get_experiment_results(exp_groups=exp_groups)

    # Save formatted results
    save_formatted_results_question(results_question, output_file="results/results_pilot2ndrun_question.json")
    print("Question wise Results have been saved to results/results_pilot2ndrun_question.json")
    save_formatted_results_vis(results_vis,output_file="results/results_pilot2ndrun_vis.json")
    print("Dataset wise Results have been saved to results/results_pilot2ndrun_vis.json")
    save_formatted_results_skills(results_skill,output_file="results/results_pilot2ndrun_skill.json")
    print("Question-Skill wise Results have been saved to results/results_pilot2ndrun_skill.json")