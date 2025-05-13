import os
import json
import argparse
from src.eval_generator.dataset_loader import DatasetLoader
from src.eval_generator.client import GPT4oClient
from src.eval_generator.utils import read_questions_json

def main(csvs_directory, jsons_directory, api_key):
    gpt4o_client = GPT4oClient(api_key)
    target_subdirs = {str(i) for i in range(404,701) if i!="593"} #Add your target subdirectories here  #None: 402,569,594,559,422,688,478,675,672,504561,595  #No csv:593

    for subdir in os.listdir(csvs_directory):
        if subdir in target_subdirs:
            csv_subdir_path = os.path.join(csvs_directory, subdir)
            json_subdir_path = os.path.join(jsons_directory, subdir)
            goal_file_path = os.path.join(json_subdir_path, 'goal.json')
            # Check if the goal.json file already exists
            if os.path.exists(goal_file_path):
                print(f"Goal file already exists for directory: {csv_subdir_path}, skipping...")
                continue
            if os.path.isdir(csv_subdir_path):
                print(f"Processing directory: {csv_subdir_path}")
                
                dataset_loader = DatasetLoader(csv_subdir_path, json_subdir_path)
                dataset_summary = dataset_loader.get_summary()
                # print(dataset_summary)

                # Read Questions
                questions = read_questions_json(json_subdir_path)
                # print(questions)
                
                persona_goal = gpt4o_client.generate_persona_goal(questions, dataset_summary)
                print(persona_goal)
                
                # Save the best questions to a JSON file
                output_file = os.path.join(json_subdir_path, 'goal.json')
                with open(output_file, 'w') as f:
                    json.dump(persona_goal, f, indent=4)
                print(f"Saved goal: {output_file}")
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation sets for data analytics agents.")
    parser.add_argument('--csvs_directory', type=str, help='The directory containing subfolders data csvs.')
    parser.add_argument('--jsons_directory', type=str, help='The directory containing subfolders meta information.')
    parser.add_argument('--api_key', type=str, help='The API key for OpenAI.')
    
    args = parser.parse_args()
    main(args.csvs_directory, args.jsons_directory, args.api_key)