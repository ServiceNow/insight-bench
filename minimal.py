import os
from insightbench import benchmarks, agents

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "<openai_api_key>"

# Get Dataset
dataset_csv_path = "data/notebooks/csvs/flag-1.csv"
dataset_notebook_path = "data/notebooks/flag-1.ipynb"
dataset_dict = benchmarks.load_dataset_dict(
    dataset_csv_path=dataset_csv_path, dataset_notebook_path=dataset_notebook_path
)
# Run an Agent
agent = agents.Agent(
    model_name="gpt-4o-mini",
    max_questions=2,
    branch_depth=1,
    n_retries=2,
    savedir="results/sample",
)
pred_insights = agent.get_insights(dataset_csv_path=dataset_csv_path)

# Evaluate
score = benchmarks.evaluate_insights(
    pred=pred_insights, gt=dataset_dict["insights"], method="rouge"
)

# Print Score
print("score: ", score)
