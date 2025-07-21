import os
import json
import argparse
from src.eval_generator.dataset_loader import DatasetLoader
from src.eval_generator.notebook_processor import NotebookProcessor
from src.eval_generator.client import GPT4oClient
from data.tasks_and_skills import SKILLS, TASKS
from src.eval_generator.utils import check_task_and_skill
from src.eval_generator.rag_answer_validator import RAGAnswerValidator


def main(csvs_directory, jsons_directory, notebooks_directory, api_key):
    gpt4o_client = GPT4oClient(api_key)
    rag_answer_validator = RAGAnswerValidator(gpt4o_client)

    for subdir in os.listdir(csvs_directory):
        csv_subdir_path = os.path.join(csvs_directory, subdir)
        json_subdir_path = os.path.join(jsons_directory, subdir)
        notebook_subdir_path = os.path.join(notebooks_directory, subdir)
        if os.path.isdir(csv_subdir_path):
            print(f"Processing directory: {csv_subdir_path}")

            # Load the dataset and generate summary
            dataset_loader = DatasetLoader(csv_subdir_path, json_subdir_path)
            dataset_summary = dataset_loader.get_summary()

            # Process the notebook and create cell batches
            notebook_processor = NotebookProcessor(notebook_subdir_path)
            cell_batches = notebook_processor.process_notebook()

            questions = []
            for batch in cell_batches:
                cell_numbers = [cell_number for cell_number, _ in batch]
                batch_questions = gpt4o_client.generate_questions(
                    batch, dataset_summary, SKILLS, TASKS
                )
                for batch_question in batch_questions:

                    if not check_task_and_skill(
                        batch_question["task"], batch_question["skill"]
                    ):
                        continue
                    if not rag_answer_validator.check_answer(
                        batch_question["question"], batch_question["answer"], batch
                    ):
                        continue

                    questions.append(batch_question)

            # Select the 10 best questions
            best_questions = gpt4o_client.select_best_questions(questions)

            # Save the best questions to a JSON file
            output_file = os.path.join(json_subdir_path, "questions.json")
            with open(output_file, "w") as f:
                json.dump(best_questions, f, indent=4)
            print(f"Saved evaluation set to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation sets for data analytics agents."
    )
    parser.add_argument('csvs_directory', type=str, help='The directory containing subfolders data csvs.')
    parser.add_argument('jsons_directory', type=str, help='The directory containing subfolders meta information.')
    parser.add_argument('notebooks_directory', type=str, help='The directory containing subfolders notebooks.')
    parser.add_argument("api_key", type=str, help="The API key for OpenAI.")

    args = parser.parse_args()
    main(args.csvs_directory, args.jsons_directory, args.notebooks_directory, args.api_key)