from pathlib import Path
import json


def get_all_skills(dataset_path: Path | str) -> set:
    """
    Retrieves a set of all unique skills from 'questions.json' files in dataset subdirectories.
    Args:
        dataset_path: Path to the datasets directory
    Returns:
        set: A set of skill names from all questions
    """
    # Convert string path to Path object if needed
    dataset_path = Path(dataset_path)
    skills = set()

    # Check if directory exists
    if not dataset_path.exists():
        print(f"Warning: Directory {dataset_path} does not exist")
        return skills

    # Iterate through all subdirectories
    for skill_dir in dataset_path.iterdir():
        if skill_dir.is_dir():
            questions_file = skill_dir / "questions.json"
            if questions_file.exists():
                try:
                    with questions_file.open("r", encoding="utf-8") as f:
                        questions = json.load(f)
                        # Extract skills from each question
                        for question in questions:
                            if "skill" in question:
                                skills.add(question["skill"])
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {questions_file}")
                except Exception as e:
                    print(f"Error processing {questions_file}: {str(e)}")

    return skills


if __name__ == "__main__":
    # Test the function with default path
    default_path = Path("data/datasets")
    skills = get_all_skills(default_path)
    print(f"Found {len(skills)} skills: {sorted(skills)}")
