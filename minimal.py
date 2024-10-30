import os
from insightbench import benchmarks, agents

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "<openai_api_key>"


# Get Dataset
dataset_dict = benchmarks.load_dataset_dict("data/notebooks/flag-1.json")

# Run an Agent
agent = agents.Agent(
    model_name="gpt-4o-mini",
    max_questions=2,
    branch_depth=1,
    n_retries=2,
    savedir="results/sample",
)
pred_insights, pred_summary = agent.get_insights(
    dataset_csv_path=dataset_dict["dataset_csv_path"], return_summary=True
)


# Evaluate
score_insights = benchmarks.evaluate_insights(
    pred_insights=pred_insights,
    gt_insights=dataset_dict["insights"],
    score_name="rouge1",
)
score_summary = benchmarks.evaluate_summary(
    pred=pred_summary, gt=dataset_dict["summary"], score_name="rouge1"
)

# Print Score
print("score_insights: ", score_insights)
print("score_summary: ", score_summary)
