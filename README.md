# Insight-Bench

![Banner](data/banner.jpg)

## Evaluating Data Analytics Agents Through Multi-Step Insight Generation
[[Paper]](https://insightbench.github.io/)[[Website]](https://insightbench.github.io/)[[Dataset]](https://huggingface.co/datasets/ServiceNow/insight_bench)


Insight-Bench is a benchmark dataset designed to evaluate end-to-end data analytics by evaluating agents' ability to perform comprehensive data analysis across diverse use cases, featuring carefully curated insights, an evaluation mechanism based on LLaMA-3-Eval or G-EVAL, and a data analytics agent, AgentPoirot.

## Data

All groundtruth notebooks are in `data/notebooks`. 

An example notebook can be found here: `data/notebooks/flag-1.ipynb`

## 1. Install the python libraries

```
pip install --upgrade git+https://github.com/ServiceNow/insight-bench
```

## 2. Usage

Evaluate agent on a single notebook

```python
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
```

## 3. Evaluate Agent on Multiple Insights

```bash
python main.py --openai_api_key <openai_api_key>
               --savedir_base <savedir_base>
```
## 4. Running AgentADA

```bash
cd  ada_insightarena
python main_insightarena.py -e insights_insightarena
```
## Citation

```bibtex
@article{sahu2024insightbench,
  title={InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation},
  author={Sahu, Gaurav and Puri, Abhay and Rodriguez, Juan and Abaskohi, Amirhossein and Chegini, Mohammad and Drouin, Alexandre and Taslakian, Perouz and Zantedeschi, Valentina and Lacoste, Alexandre and Vazquez, David and Chapados, Nicolas and Pal, Christopher and others},
  journal={arXiv preprint arXiv:2407.06423},
  year={2024}
}

```

## 🤝 Contributing
- Please check the outstanding issues and feel free to open a pull request.
- Please include any feedback or suggestions or feature requests in the issues section.
- You are welcome to contribute to the codebase and add new datasets and flags


### Thank you!