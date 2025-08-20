import json
import os
from src import utils
from tqdm import tqdm
from openai import OpenAI

llm_client = OpenAI()
exp_dict = {
    "challenge": "test_20",
    "model": "gpt-4o",
    "eval_mode": "insights",
    "with_skills": 0,
    "insight_gen": "modified_insight_gen",
    "prompt": "amrutha"
}
exp_group = 'insights_w_skills'
exp_hash = utils.get_exp_hash(exp_dict)
savedir = f"results/{exp_group}/{exp_hash}"
print(savedir)
all_ids = list(range(49,59))
answer_list = []
base_path = 'data/jsons'
# print(all_ids)
for id in tqdm(all_ids):
    with open(f'results/insights_w_skills/ac16529ea7a52a36b16324df8bb075bf/{str(id)}/ques_ans.json', 'r') as f:
        ques_ans = json.load(f)
        for item in ques_ans:
            answer_list.append({'question':item['question'],'answer':item['predicted_insight']})
    
    meta_dict = json.load(open(f"{base_path}/{str(id+1)}/meta.json", "r"))
    goal_dict = json.load(open(f"{base_path}/{str(id+1)}/goal.json", "r"))
    question_list = json.load(open(f"{base_path}/{str(id+1)}/questions.json", "r"))

    data_dict = {}
    data_dict["id"] = id+1
    data_dict["questions"] = question_list
    data_dict["dataset_path"] = f"data/csvs/{str(id+1)}/data.csv"
    data_dict["meta"] = meta_dict
    data_dict["goal"] = goal_dict["goal"]
    data_dict["persona"] = goal_dict["persona"]
    # data_dict["insight"] = insight_dict["insight"]
    data_dict["table"] = utils.check_and_fix_dataset(
        f"data/csvs/{id}/data.csv"
    )
    
    insight_prompt = f"""I have the following information:
    1. **Dataset Description**: {meta_dict['dataset_description']}
    2. **Analysis Goal**: {goal_dict["goal"]}
    3. **Questions and Answers**: 
    {answer_list}

    {prompt} 

    """

    system_prompt = """
    You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
    You have access to a team of highly skilled data scientists that can answer complex questions about the data.
    You call the shots and they do the work.
    Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
    """
    response = llm_client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role":"system", "content": system_prompt},{"role": "user", "content": insight_prompt}],
        temperature=0.0,
        ).choices[0].message.content
    # visualize the insights
    vis_insights(
        pred_insights=response,
        gt_insights='',
        data_dict=data_dict,
        savedir=os.path.join(savedir, f"vis_{id}"),
    )

