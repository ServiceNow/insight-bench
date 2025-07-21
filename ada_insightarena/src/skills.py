# from skill_exemplers import kmeans, arima
from urllib import response
from openai import OpenAI
import os
from dotenv import load_dotenv

from pathlib import Path
import difflib
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from src.utils import get_llm_response, get_llm_response_with_schema
from pydantic import BaseModel

# Load environment variables at the start
load_dotenv()

llm_client = OpenAI()


class Filename(BaseModel):
    file_name: str


class FileRelevanceSchema(BaseModel):
    names: list[Filename]


def get_embeddings(model_name, embed_input):
    from sentence_transformers import SentenceTransformer, util

    """Create embeddings using either sentence transformers or openai embeddings"""
    # Check if model_name is 'openai'
    if model_name == "openai":
        response = llm_client.embeddings.create(
            input=embed_input, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    # Otherwise use sentence transformers
    model = SentenceTransformer(model_name)
    embedding = model.encode(embed_input, convert_to_tensor=True)
    return embedding


def get_documentation_with_bert(
    skill_name: str, question: str, summary_flag: bool = True
) -> list:
    """
    Retrieve the top 3 matching markdown files using embeddings and cosine similarity.
    Matches are based on both skill name and question context.
    """
    from sentence_transformers import SentenceTransformer, util

    if summary_flag:
        docs_path = Path("data/skills/algorithms_summary")
        all_files = list(docs_path.glob("*.txt"))
    else:
        docs_path = Path("data/skills/algorithms")
        # Get all markdown files
        all_files = list(docs_path.glob("*.md"))

    embedding_model_name = "openai"  # "all-MiniLM-L6-v2"  # openai
    if not all_files:
        return []

    # Generate embedding for the skill name
    skill_embedding = get_embeddings(embedding_model_name, skill_name)
    question_embedding = get_embeddings(embedding_model_name, question)

    def process_file(file_path, skill_embedding, question_embedding):
        content = file_path.read_text()
        content_embedding = get_embeddings(embedding_model_name, content)

        # Calculate similarities for both skill and question
        skill_similarity = util.pytorch_cos_sim(
            skill_embedding, content_embedding
        ).item()
        question_similarity = util.pytorch_cos_sim(
            question_embedding, content_embedding
        ).item()
        # Combine similarities with weights (adjust weights as needed)
        combined_similarity = (skill_similarity * 0.2) + (question_similarity * 0.8)

        return (file_path, combined_similarity)

    # Create partial function with both embeddings
    process_file_with_embeddings = partial(
        process_file,
        skill_embedding=skill_embedding,
        question_embedding=question_embedding,
    )

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        matches = list(executor.map(process_file_with_embeddings, all_files))

    # Sort and get top 3 matches
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:3]

    results = []
    for file_path, confidence in top_matches:
        results.append(
            {
                "file_name": file_path.stem,
                "confidence": round(confidence * 100, 2),
            }
        )

    return results


def get_documentation(skill_name: str) -> list:
    """Retrieve the top 3 matching markdown files for a skill with confidence scores."""
    docs_path = Path("data/skill_algorithms")

    # Get all markdown files
    all_files = list(docs_path.glob("*.md"))

    if not all_files:
        return []

    # Calculate similarity scores for each file
    matches = []
    for file_path in all_files:
        # Read the content of the file
        content = file_path.read_text()

        # Calculate similarity based on both filename and content
        filename_ratio = difflib.SequenceMatcher(
            None, skill_name.lower(), file_path.stem.lower()
        ).ratio()
        content_ratio = difflib.SequenceMatcher(
            None, skill_name.lower(), content.lower()
        ).ratio()

        # Use weighted average (giving more weight to content)
        combined_ratio = (filename_ratio * 0.1) + (content_ratio * 0.9)
        matches.append((file_path, combined_ratio))

    # Sort by confidence score and get top 3
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:3]

    # Read content for top matches
    results = []
    for file_path, confidence in top_matches:
        results.append(
            {
                "file_name": file_path.stem,
                "confidence": round(confidence * 100, 2),
            }
        )
    return results


def get_documentation_with_openai(
    skill_name: str, question: str, summary_flag: bool = True
) -> list:
    """
    Retrieve the top 3 matching files using OpenAI to analyze relevance between the question
    and available documentation.

    Args:
        skill_name: The name of the skill to search for
        question: The user's question about the skill
        summary_flag: If True, search in summary files, else in full documentation

    Returns:
        list: Top 3 matching files with confidence scores
    """
    # Set up the correct path based on summary flag
    docs_path = Path(
        "data/skills/algorithms_summary" if summary_flag else "data/skills/algorithms"
    )
    all_files = list(docs_path.glob("*.txt" if summary_flag else "*.md"))

    if not all_files:
        return []

    # Create a list of all document contents with their file names
    documents = []
    for file_path in all_files:
        content = file_path.read_text()
        documents.append({"name": file_path.stem, "content": content})

    prompt = f"""Given a question about a skill and several documentation files, identify the top 3 most relevant files to solve the question.

    Question: {question}

    Available documentation files:
    {json.dumps([doc['name'] for doc in documents], indent=2)}

    For each file, analyze its relevance to the question and skill, and return the top 3 files in the decreasing order of usefulness. The output should be in JSON format like this:
    [
        {{"file_name": "most_relevant_file"}},
        {{"file_name": "second_relevant_file"}},
        {{"file_name": "third_relevant_file"}}
    ]
    """

    # Get OpenAI's response
    # response = llm_client.chat.completions.create(
    #     model="gpt-4-turbo-preview",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that analyzes document relevance.",
    #         },
    #         {"role": "user", "content": prompt},
    #     ],
    #     response_format={"type": "json_object"},
    # )

    try:
        # Parse the response and extract the matches
        result = get_llm_response_with_schema(prompt, FileRelevanceSchema)
        result = json.loads(result.choices[0].message.content)
        return result["names"]
        # result = json.loads(response.choices[0].message.content)
        # return result.get("matches", [])[:3]  # Ensure we return at most 3 matches
    except (json.JSONDecodeError, KeyError, AttributeError):
        # Return empty list if there's any error in parsing
        return []


def get_skill(question: str, skill_name: str, task_name: str = None):
    if skill_name == "kmeans":
        return kmeans.kmeans_skill
    elif skill_name == "auto":
        # skill_doc_with_bert = get_documentation_with_bert(
        #     task_name, question, summary_flag=True
        # )
        # return skill_doc_with_bert
        skill_doc_with_openai = get_documentation_with_openai(
            task_name, question, summary_flag=True
        )
        return skill_doc_with_openai
    elif skill_name == "arima":
        return arima.arima_skill
    elif skill_name == "":
        return ""
    else:
        raise ValueError(f"Skill {skill_name} not found")


def get_skills_from_questions(questions_list) -> list:
    """Retrieve all questions and their detected skills from questions.json.

    Args:
        questions_file (str): Path to the input questions JSON file
        skills_file (str): Path to the output skills JSON file

    Returns:
        list: List of questions with their detected skills
    """
    questions_with_skills = []  # Initialize the list to store all questions

    # Iterate through the dictionary where keys are file paths

    for q_dict in tqdm(questions_list, desc="Processing questions"):
        question = q_dict["question"]
        task = q_dict["task"]

        if task == "Basic Data Analysis":
            # Special case for Basic data analysis
            retrieved_doc = [{"file_name": "Basic Data Analysis", "confidence": 1.0}]
        else:
            # Get skill using get_skill function with task_name
            retrieved_doc = get_skill(
                question,
                "auto",
                task,
            )

        question_data = {
            "question": question,
            "task": task,
            "skill_notebooks": retrieved_doc,
            "predicted_skills": [doc["file_name"] for doc in retrieved_doc],
        }
        questions_with_skills.append(question_data)
    return questions_with_skills


def main():
    questions_file = "results/questions.json"
    skills_file = "results/skills.json"
    get_skills_from_questions(questions_file, skills_file)
    print(f"Saved questions and skills analysis to {skills_file}")


if __name__ == "__main__":
    main()
