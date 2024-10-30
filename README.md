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
dataset_csv_path = "data/datasets/flag-1.csv"
dataset_notebook_path = "data/notebooks/flag-1.ipynb"
dataset_dict = benchmarks.load_dataset_dict(dataset_csv_path=dataset_csv_path, 
                                            dataset_notebook_path=dataset_notebook_path)
# Run an Agent
agent = agents.Agent(model_name="gpt-4o-mini", max_questions=10, branch_depth=2, n_retries=2, savedir="results/sample")
pred_insights = agent.get_insights(dataset_csv_path=dataset_csv_path)

# Evaluate
score = evaluate_insights(pred=pred_insights, gt=dataset_dict['insights'])

# Print Score
print("score: ", score)
```

## 3. Evaluate Agent on Multiple Insights

```bash
python main.py --openai_api_key <openai_api_key>
               --savedir_base <savedir_base>
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