from data.tasks_and_skills import SKILLS
import os
import json


def check_task_and_skill(task, skill):
    if task not in SKILLS:
        print(f"Invalid task: {task}")
        return False
    if skill not in SKILLS[task]:
        print(f"Invalid skill: {skill}")
        return False

    return True


def read_questions_json(subdir_path):
    file_path = os.path.join(subdir_path, "questions.json")
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        directory_id = os.path.basename(subdir_path)  # Assuming the directory name is the ID
        print(f"Skipping directory {directory_id} due to missing or empty questions.json")
        return None  # Return None to indicate that the file should be skipped

    with open(file_path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON content in questions.json of {directory_id}: {e}")
            return None  # Return None to indicate that the file should be skipped