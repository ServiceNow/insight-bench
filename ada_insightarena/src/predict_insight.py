import openai
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from contextlib import redirect_stdout
import os
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import hashlib, json

import base64

import markdown
from bs4 import BeautifulSoup
import traceback

from src.utils import get_llm_response, get_llm_response_with_schema, load_prompt
from pydantic import BaseModel

llm_client = OpenAI()
generic_insight_flag = 1


class Filename(BaseModel):
    file_name: str


class FileRelevanceSchema(BaseModel):
    names: list[Filename]

def markdown_to_text(markdown_file_path):
    # Read the Markdown file
    with open(markdown_file_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    # Convert Markdown content to HTML
    html_content = markdown.markdown(markdown_content)

    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(html_content, "html.parser")
    text_content = soup.get_text()

    return text_content


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gen_analytics_code_plot(
    question: str,
    table: pd.DataFrame,
    skill="",
    skill_exempler="",
    savedir="",
    skill_flag=0,
    ques_no=0,
    model="gpt-4o",
    summary_flag=0,
    dataset_description=None
) -> str:
    """
    Generate and execute code to create a plot based on a data analytics question.

    Args:
        question (str): The data analytics question to be answered
        df (pd.DataFrame): Sample dataframe containing the data

    Returns:
        str: Path to the saved plot image
    """
    global df
    df = table
    savedir = os.path.join(savedir, "question_" + str(ques_no))
    os.makedirs(savedir, exist_ok=True)

    # Create context for the LLM

    # Providing dynamic DataFrame context
    df_info = "DataFrame has columns such as: " + ", ".join(
        [f"{col} (Type: {df[col].dtype})" for col in df.columns]
    )
    if dataset_description is None:
        df_description = "This DataFrame includes various columns, which could represent different data types and sectors, such as numerical, categorical, or datetime values, depending on the specific dataset provided."
    else:
        df_description = dataset_description
    df_head = f"First few rows:\n{df.head().to_string()}"
    # without Skill Exempler
    if skill_flag == 0 or skill_exempler == "":
        print('In No Skill-------------------------------------')
        code_gen_prompt_inputs =  {
            'df_info': df_info,
            'df_description': df_description,
            'df_head': df_head,
            'question': question,
            'savedir': savedir
            }
        prompt = load_prompt('code_gen_without_skills.txt', **code_gen_prompt_inputs)

    # With Skill Exempler
    elif skill_flag == 1:
        if summary_flag == 0:
            summary_prompt_inputs =  {'skill_exempler': skill_exempler}
            summary_prompt = load_prompt('skill_summart.txt', **summary_prompt_inputs)
            # Call the LLM to summarize the skill exemplar
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.0,
            )

            # Extract the summary from the response
            skill_exempler_summary = response.choices[0].message.content
            with open(os.path.join(savedir, "skill_exempler_summary.txt"), "w") as f:
                f.write(summary_prompt)
        else:
            skill_exempler_summary = skill_exempler

        # with open(os.path.join(savedir, "planned_summary.txt"), "w") as f:
        #     f.write(planned_summary)
        code_gen_prompt_inputs =  {
            'df_info': df_info,
            'df_description': df_description,
            'df_head': df_head,
            'skill_exempler_summary': skill_exempler_summary,
            'question': question,
            'savedir': savedir
            }
        prompt = load_prompt('code_gen_with_skills.txt', **code_gen_prompt_inputs)
       
    # local_vars = {"df": df, "plt": plt}
    try:
        skill_check_flag = False
        for i in range(3):
            try:
                # Get response from LLM
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                # Save prompt to file
                with open(os.path.join(savedir, "prompt.txt"), "w") as f:
                    f.write(prompt)

                # Get the generated code and extract code between triple backticks
                full_response = response.choices[0].message.content
                code_blocks = full_response.split("```")

                if len(code_blocks) >= 3:
                    # Extract the code from between the backticks (second element)
                    generated_code = code_blocks[1]
                    # Remove "python" if it's at the start of the code
                    generated_code = generated_code.replace("python\n", "", 1)
                else:
                    # If no code blocks found, try to use the full response
                    generated_code = full_response
                # Execute the generated code with df in locals
                # local_vars['print'] = lambda *args, **kwargs: output_log.append(' '.join(map(str, args)))
                predicted_skill = get_skill_from_generated_code(generated_code)
                if skill_flag == 1 and skill_check_flag==False:
                    if predicted_skill!=skill+'.txt':
                        print('---------------Skill:', skill)
                        print('---------------Predicted Skill:', predicted_skill)
                        prompt += f''' \n Make sure to use the given **skill_exemplar**: {skill} - {skill_exempler_summary}\n 
                        The generated_code is {generated_code}'''
                        with open(os.path.join(savedir, "error_skill.txt"), "w+") as f:
                            f.write(f'The generated code doesnt use the required skill \n {generated_code}')
                        skill_check_flag = True
                        continue
                exec(generated_code, globals())  # local_var
                # captured_output = '\n'.join(output_log)
                # print(captured_output)
                break
            except Exception as e:
                with open(os.path.join(savedir, "error.txt"), "w+") as f:
                    f.write(str(e))
                error_prompt = f"""\n\nGiven the generated code and the error debug the code and return the output code according to the **Output Instructions**.
                The generated code is: {generated_code}
                The error is: {str(e)}"""
                prompt += error_prompt
                continue

        # Save the generated code to code.py
        with open(os.path.join(savedir, "code.py"), "w") as f:
            f.write(generated_code)
        # Ensure the plot is saved with the correct filename
        plot_path = os.path.join(savedir, "plot.jpeg")
        plt.savefig(plot_path)
        plt.close()

        answer_prompt_inputs = {"question": question, "df_head": df_head, "stats": stats}
        answer_prompt = load_prompt('answer_generation.txt', **answer_prompt_inputs)

        base64_image = encode_image(plot_path)
        answer_response = (
            llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a highly skilled data analyst with expertise in interpreting complex visualizations and uncovering actionable insights from data.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": answer_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
            .choices[0]
            .message.content
        )
        
        summarized_insight_response = summarize_answer(question, answer_response, model=model)

        return plot_path, summarized_insight_response, predicted_skill

    except Exception as e:
        error_message = f"Error executing generated code: {str(e)}\nGenerated code was:\n{generated_code}"
        print(error_message)

        # Save error to file
        with open(os.path.join(savedir, "error.txt"), "w") as f:
            f.write(error_message)

        return None, None


def predict_insight_categories(
    questions_list, dataset_description, goal, model="gpt-4o"
):
    """
    Predict the top 3 categories where insights from the answers can be categorized.

    Args:
        questions_list (list): List of questions that were analyzed
        dataset_description (str): Description of the dataset
        goal (str): Analysis goal
        model (str): LLM model to use

    Returns:
        list: Top 3 categories for insight categorization
    """
    category_prompt_inputs = {"dataset_description": dataset_description, "goal": goal, "questions_list": questions_list}
    category_prompt = load_prompt('insight_category_prediction.txt', **category_prompt_inputs)

    system_prompt = """
    You are an advanced data scientist specializing in **predicting insight structures before analysis is completed**.
    Your expertise lies in **anticipating the nature of insights** that will be extracted and organizing them into **optimal, high-value categories** for business decision-making.
    Focus on identifying **the most meaningful, distinct, and action-driven categories** to ensure insights are structured effectively.
    """

    response = (
        llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": category_prompt},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message.content
    )

    return response

def summarize_answer(question, answer, model="gpt-4o"):

    summarize_answer_prompt_inputs = {'answer': answer, 'question':question}
    summarize_answer_prompt = load_prompt('answer_summarize.txt', **summarize_answer_prompt_inputs)
    response = (
        llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": summarize_answer_prompt},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message.content
    )

    return response


def get_analytical_insight(
    insight_categories, answer_list, dataset_description, goal, model="gpt-4o"
):
    """
    Get the insight based on the list of answers
    """

    insight_prompt_inputs = {"dataset_description":dataset_description, "goal":goal, "answer_list":answer_list, "insight_categories": insight_categories}
    insight_prompt = load_prompt('insight_generation.txt', **insight_prompt_inputs)

    system_prompt = """
    You are an expert data science manager specializing in **distilling complex analyses into concise, engaging, and actionable insights**.
    Your expertise lies in **extracting high-value insights that drive strategic decisions**, ensuring that each insight is **relevant, data-driven, and practical**.
    Your final deliverable is a **visually engaging, structured report** that highlights the most critical findings, integrates supporting data naturally, and provides actionable steps for business impact.
    If a highly impactful insight does not fit any existing category, you must **intelligently define a new category** that adds unique value without redundancy.
    """

    response = (
        llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": insight_prompt},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message.content
    )

    return response

def get_skill_from_generated_code(code):
    '''
    Get the skill used for solving the question from the generated code from LLM
    '''

    list_of_skills = os.listdir('data/skills/algorithms_summary')

    skill_from_generated_code_inputs = {"code":code, "list_of_skills":list_of_skills}
    skill_from_generated_code_prompt = load_prompt('predict_skill_generated_code.txt', **skill_from_generated_code_inputs)

    response = get_llm_response_with_schema(skill_from_generated_code_prompt, FileRelevanceSchema)
    predicted_skill = json.loads(response.choices[0].message.content)
    return predicted_skill['names'][0]['file_name']

if __name__ == "__main__":
    print("in main")
    # for ques, skill in ques.json:
    path = "../data/skills/algorithms/GrangerCausality.md"
    # with open(path, 'r') as file:
    #     skill_exempler = file.read()
    plain_text = markdown_to_text(path)
    # print(plain_text)
    question = "What does the Granger causality test reveal about the relationship between Sales and TV Ad Budget?"
    df = pd.read_csv("../data/csvs/6/data.csv")
    savedir = "results/6"
    skill_exempler = plain_text
    os.makedirs(savedir, exist_ok=True)
    skill_flag = 1
    plot_path, insight_response = gen_analytics_code_plot(
        question, df, skill_exempler, savedir, skill_flag
    )  # save insights
    print("plot_path", plot_path)
    print("insight_response", insight_response)
    # print('quantitative_results', quantitative_results)
