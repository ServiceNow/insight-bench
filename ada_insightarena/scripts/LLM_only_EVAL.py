import json
import os

from openai import OpenAI

client = OpenAI()


def evaluate_insights_comparison(
    dataset_id,
    with_skills_hash,
    without_skills_hash,
    persona="You are a helpful AI assistant",
):
    """
    Compare insights from two different runs (with and without skills) for a given dataset
    and evaluate their relevance to the goal.

    Args:
        dataset_id (str): The dataset identifier
        with_skills_hash (str): Hash value for the run with skills
        without_skills_hash (str): Hash value for the run without skills
        persona (str): The persona to use for evaluation

    Returns:
        tuple: (evaluation_prompt, llm_response)
    """
    # Read the goal
    goal_path = f"data/jsons/{dataset_id}/goal.json"
    try:
        with open(goal_path, "r") as f:
            goal_data = json.load(f)
            goal = goal_data.get("goal", "")
    except FileNotFoundError:
        raise Exception(f"Goal file not found at {goal_path}")

    # Read insights with skills
    with_skills_path = (
        f"results/insights_w_skills/{with_skills_hash}/{dataset_id}/final_insight.txt"
    )
    try:
        with open(with_skills_path, "r") as f:
            with_skills_insight = f.read().strip()
    except FileNotFoundError:
        raise Exception(f"With-skills insight file not found at {with_skills_path}")

    # Read insights without skills
    without_skills_path = f"results/insights_wo_skills/{without_skills_hash}/{dataset_id}/final_insight.txt"
    try:
        with open(without_skills_path, "r") as f:
            without_skills_insight = f.read().strip()
    except FileNotFoundError:
        raise Exception(
            f"Without-skills insight file not found at {without_skills_path}"
        )

    # Construct the evaluation prompt
    evaluation_prompt = f"""{persona}

Please evaluate and compare the following two insights in terms of their relevance to the given goal. 
For each criterion below, distribute 1 point between the two insights based on their relative performance.
For example, if Insight 1 performs slightly better than Insight 2 on a criterion, you might assign 0.6 to Insight 1 and 0.4 to Insight 2.
The scores for each criterion MUST sum to 1.

Criteria:
1. Depth of Analysis (How deep is the analysis?)
2. Relevance to Goal (How well does it address the main objective?)
3. Persona Consistency (How well does it align with the persona?)
4. Coherence (How coherent and cohesive is the analysis?)
5. Answers Question Adequately (How well does it answer the question?)
6. Plot Conclusion (How well does it briefly summarize the plot?)

Goal:
{goal}

Persona: {persona}

Insight 1 (With Skills):
{with_skills_insight}

Insight 2 (Without Skills):
{without_skills_insight}

Please provide:
1. For each criterion, assign scores that sum to 1 between the two insights
   Format: "Criterion Name: score_1 – brief explanation | score_2 – brief explanation"
2. A brief explanation for each score
3. A final comparison highlighting the key differences between the two insights
4. A recommendation on which insight better serves the goal and why

Remember: For each criterion, the two scores MUST sum to 1."""

    # Get response from LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or your preferred model
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.0,
        )
        llm_response = response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting LLM response: {str(e)}")

    return llm_response


def get_all_dataset_ids(with_skills_hash, without_skills_hash):
    """
    Get all dataset IDs that exist in both with_skills and without_skills directories.
    Only returns IDs that are purely numerical.
    """
    with_skills_path = f"results/insights_w_skills/{with_skills_hash}"
    without_skills_path = f"results/insights_wo_skills/{without_skills_hash}"

    # Get datasets that exist in both directories
    with_skills_datasets = (
        set(os.listdir(with_skills_path)) if os.path.exists(with_skills_path) else set()
    )
    without_skills_datasets = (
        set(os.listdir(without_skills_path))
        if os.path.exists(without_skills_path)
        else set()
    )

    # Filter for numerical-only IDs
    common_datasets = with_skills_datasets.intersection(without_skills_datasets)
    numerical_datasets = {
        dataset_id for dataset_id in common_datasets if dataset_id.isdigit()
    }

    return list(numerical_datasets)


def main():
    # Example usage
    with_skills_hash = "pilot"
    without_skills_hash = "pilot"

    try:
        dataset_ids = get_all_dataset_ids(with_skills_hash, without_skills_hash)
        # dataset_ids = list(range(50, 52))
        if not dataset_ids:
            raise Exception("No matching datasets found in both directories")

        for dataset_id in dataset_ids:
            print(f"\nProcessing dataset: {dataset_id}")
            llm_response = evaluate_insights_comparison(
                dataset_id, with_skills_hash, without_skills_hash
            )

            # Create the output directory if it doesn't exist
            output_dir = f"results/evaluations/GPTEval/pilot/{dataset_id}"
            os.makedirs(output_dir, exist_ok=True)

            # Save the LLM response to files

            llm_response_path = f"{output_dir}/llm_response.txt"
            with open(llm_response_path, "w") as f:
                f.write(llm_response)
            print(f"LLM response saved to: {llm_response_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
