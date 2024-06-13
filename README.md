# Insight-Bench

## Evaluating Business Analytics Agents Through Multi-Step Insight Generation
[[Website]](https://insightbench.github.io/)[[Dataset]](https://huggingface.co/datasets/ServiceNow/insight_bench)


Insight-Bench is a benchmark dataset designed to evaluate end-to-end data analytics by evaluating agents' ability to perform comprehensive data analysis across 31 diverse business use cases, featuring carefully curated insights, an evaluation mechanism based on LLaMA-3-Eval, and a data analytics agent, AgentPoirot


## Pre-requisites

- Install the python libraries
```
pip install .
```

- Specify the OpenAI key in your environment
```
export OPENAI_API_KEY="your-api-key"
```

All groundtruth notebooks are in `data/notebooks`. 

An example notebook can be found here: data/notebooks/flag-1.ipynb

## Quick Start

Run the following command to run AgentPoirot on one of the notebook flags

```
python run_agent.py -e <exp_group> -sb <savedir_base>
```

The variables <...> can be substituted with the following values:

- <exp_group> : quick
- <savedir_base>:  path to where results will be saved

Experiment hyperparameters are defined in `exp_groups.py`



