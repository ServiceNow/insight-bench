import openai
import pandas as pd
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
import numpy as np
from contextlib import redirect_stdout
import os
import shutil
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from src.predict_questions import (
    generate_analytics_questions,
    generate_analytics_questions_with_goal_persona,
    generate_analytics_questions_iterative_with_goal_persona
)
from src.skills import get_skills_from_questions
from src.predict_insight import (
    markdown_to_text,
    gen_analytics_code_plot,
    get_analytical_insight,
    predict_insight_categories,
)
import hashlib, json
import base64
import markdown
from bs4 import BeautifulSoup
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
from src.metrics_utlis import compute_rouge_score, compute_g_eval


class Agent:

    def __init__(self, goal, persona, model, data_description, dataset_id):
        self.goal = goal
        self.persona = persona
        self.model = model
        self.data_description = data_description
        self.modernbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dataset_id = dataset_id

    def predict_insights(self, table, savedir, savedir_categories, skill_flag=1):
        """
        Predict insights for a given table, goal, and persona
        """
        # get the questions
        question_path = f'results/generated_questions/{self.dataset_id}.json'
        os.makedirs(os.path.dirname(question_path), exist_ok=True)
        if os.path.isfile(question_path):
            with open(question_path, 'r') as f:
                questions = json.load(f) 
        else:
            questions = self.get_questions_goal_persona(table)  # TODO: add persona and goal
            with open(question_path, 'w') as f:
                json.dump(questions, f, indent=4)
        # Check if insight categories file exists
        categories_file = os.path.join(savedir_categories, "insight_categories.txt")
        if os.path.exists(categories_file):
            # Load existing categories
            with open(categories_file, "r") as f:
                insight_categories = f.read()
        else:
            # Predict new categories and save them
            insight_categories = predict_insight_categories(
                questions, self.data_description, self.goal, self.model
            )
            with open(categories_file, "w") as f:
                f.write(insight_categories)
        with open(os.path.join(savedir, "just_questions.txt"), "w") as f:
            f.write(str(questions))
        questions_with_skills = self.predict_skills(questions=questions)
        # print(questions_with_skills)
        with open(os.path.join(savedir, "questions_with_skills.txt"), "w") as f:
            f.write(str(questions_with_skills))
        qa_list = []
        answer_list = []

        for i, q in tqdm(enumerate(questions_with_skills), desc="Processing answers"):
            # get the skill
            question = q["question"]
            skill = q["predicted_skills"][0]
            summary_flag = 0
            if skill_flag == 1:
                summary_path = os.path.join(
                    os.path.join("data/skills/algorithms_summary", str(skill) + ".txt")
                )
                if os.path.exists(summary_path):
                    summary_flag = 1
                    with open(summary_path, "r") as file:
                        skill_text = file.read()
                else:
                    skill_path = os.path.join("data/skills/algorithms", skill + ".md")
                    if os.path.exists(skill_path):
                        skill_text = markdown_to_text(skill_path)
                    else:
                        print("NO SKILL NAMED: ", skill)
                        skill_text = ""
            else:
                skill_text = ''
            plot,insight, code_skill = self.predict_insight(question=question, 
                table=table, 
                skill_name=skill,
                skill=skill_text, 
                savedir=savedir,
                skill_flag=skill_flag, 
                ques_no=i, 
                summary_flag=summary_flag)
            qa_list.append({'question':question, 'skill':q['predicted_skills'], 'predicted_insight':insight,'plot':plot, 'code_skill': code_skill})
            question_dir = os.path.join(savedir, "question_" + str(i))
            os.makedirs(question_dir, exist_ok=True)
            #Save the question and insight
            with open(os.path.join(question_dir, "question_insight.txt"), 'w', encoding="utf-8") as f:
                # f.write(str(insight))
                f.write(f"Question: {question}\n")
                f.write(f"Insight: {insight}\n")
            plot_dest = os.path.join(question_dir, "plot.jpeg") 
            # shutil.copy(plot, plot_dest)
            answer_list += [insight]

        # get the insights
        with open(os.path.join(savedir, "ques_ans.json"), "w") as f:
            json.dump(qa_list, f, indent=4)

        insights = self.get_insights(answer_list, insight_categories)
        with open(os.path.join(savedir, "final_insight.txt"), "w") as f:
            f.write(insights)

        return insights

    def predict_insights_wo_questions(self, table, questions, savedir, savedir_categories, skill_flag=1):
        """
        Predict insights for a given table, goal, and persona
        """
        questions = questions
        # Check if insight categories file exists
        categories_file = os.path.join(savedir_categories, "insight_categories.txt")
        if os.path.exists(categories_file):
            # Load existing categories
            with open(categories_file, "r") as f:
                insight_categories = f.read()
        else:
            # Predict new categories and save them
            insight_categories = predict_insight_categories(
                questions, self.data_description, self.goal, self.model
            )
            with open(categories_file, "w") as f:
                f.write(insight_categories)
        with open(os.path.join(savedir, "just_questions.txt"), "w") as f:
            f.write(str(questions))
        questions_with_skills = self.predict_skills(questions=questions)
        # print(questions_with_skills)
        with open(os.path.join(savedir, "questions_with_skills.txt"), "w") as f:
            f.write(str(questions_with_skills))
        qa_list = []
        answer_list = []

        for i, q in tqdm(enumerate(questions_with_skills), desc="Processing answers"):
            # get the skill
            question = q["question"]
            skill = q["predicted_skills"][0]
            summary_flag = 0
            if skill_flag == 1:
                summary_path = os.path.join(
                    os.path.join("data/skills/algorithms_summary", str(skill) + ".txt")
                )
                if os.path.exists(summary_path):
                    summary_flag = 1
                    with open(summary_path, "r") as file:
                        skill_text = file.read()
                else:
                    skill_path = os.path.join("data/skills/algorithms", skill + ".md")
                    if os.path.exists(skill_path):
                        skill_text = markdown_to_text(skill_path)
                    else:
                        print("NO SKILL NAMED: ", skill)
                        skill_text = ""
            else:
                skill_text = ''
            plot,insight, code_skill = self.predict_insight(question=question, 
                table=table, 
                skill_name=skill,
                skill=skill_text, 
                savedir=savedir,
                skill_flag=skill_flag, 
                ques_no=i, 
                summary_flag=summary_flag)
            qa_list.append({'question':question, 'skill':q['predicted_skills'], 'predicted_insight':insight,'plot':plot, 'code_skill': code_skill})
            question_dir = os.path.join(savedir, "question_" + str(i))
            os.makedirs(question_dir, exist_ok=True)
            #Save the question and insight
            with open(os.path.join(question_dir, "question_insight.txt"), 'w', encoding="utf-8") as f:
                # f.write(str(insight))
                f.write(f"Question: {question}\n")
                f.write(f"Insight: {insight}\n")
            plot_dest = os.path.join(question_dir, "plot.jpeg") 
            # shutil.copy(plot, plot_dest)
            answer_list += [insight]

        # get the insights
        with open(os.path.join(savedir, "ques_ans.json"), "w") as f:
            json.dump(qa_list, f, indent=4)

        insights = self.get_insights(answer_list, insight_categories)
        with open(os.path.join(savedir, "final_insight.txt"), "w") as f:
            f.write(insights)

        return insights

    def get_questions(self, table):
        """
        Get the questions for a given table
        """
        return generate_analytics_questions(table)

    def get_questions_goal_persona(self, table):
        """
        Get the questions for a given table
        """
        return generate_analytics_questions_iterative_with_goal_persona(
            table, self.goal, self.persona
        )
        # return generate_analytics_questions_with_goal_persona(
        #     table, self.goal, self.persona
        # )

    def get_insights(self, answer_list, insight_categories):
        """
        Get the insight based on the list of answers
        """
        return get_analytical_insight(
            insight_categories,
            answer_list,
            self.data_description,
            self.goal,
            self.model,
        )

    def predict_skills(self, table=None, questions=None):
        """
        Find the skill for a given goal and persona
        """
        if questions is None:
            questions = self.get_questions(table)

        return get_skills_from_questions(questions_list=questions)

    def predict_insight(
        self, question, table, skill_name, skill, savedir, skill_flag, ques_no, summary_flag
    ):
        """
        Find the insight for a given question
        """
        plot, insight, predicted_skill = gen_analytics_code_plot(
            question=question, 
                table=table, 
                skill=skill_name,
                skill_exempler=skill, 
                savedir=savedir,
                skill_flag=skill_flag, 
                ques_no=ques_no, 
                model=self.model, 
                summary_flag=summary_flag,
                dataset_description=self.data_description
        )

        return plot,insight, predicted_skill

    def score_insights(self, pred_insights, gt_insights, method="g-eval"):
        """
        Score the insights for a given prediction and ground truth

        method choices:
        - bleu
        - rouge
        - modernbert score
        - g-eval
        """
        if method == "bleu":
            # Using NLTK's implementation of BLEU for simplicity
            score = sentence_bleu([gt_insights.split()], pred_insights.split())
        elif method == "rouge":
            score = compute_rouge_score(pred_insights, gt_insights)
        elif method == "modernbert":
            # Compute embeddings for both the predicted and ground truth insights
            pred_embedding = self.modernbert_model.encode(
                pred_insights, convert_to_tensor=True
            )
            gt_embedding = self.modernbert_model.encode(
                gt_insights, convert_to_tensor=True
            )
            # Compute cosine similarity between embeddings
            score = util.pytorch_cos_sim(pred_embedding, gt_embedding).item()
        elif method == "g-eval":
            score = compute_g_eval(pred_insights, gt_insights, top_logprobs=5) / 10.0

        return {"insight_score": score}

    def score_skills(self, pred_skills, gt_skills, method="accuracy"):
        """
        Score the insights for a given prediction and ground truth

        method choices:
        - exact match
        - "future" Cluster Matching
        """
        if method == "accuracy":
            # Existing accuracy implementation
            matches = []
            for pred_list, gt_list in zip(pred_skills, gt_skills):
                has_match = any(pred in gt_list for pred in pred_list)
                matches.append(1.0 if has_match else 0.0)
            score = sum(matches) / len(matches) if matches else 0.0

        elif method == "mrr":
            reciprocal_ranks = []
            for pred_list, gt_list in zip(pred_skills, gt_skills):
                # Find the position of first correct prediction
                for rank, pred in enumerate(pred_list, start=1):
                    if pred in gt_list:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    # If no match found, add 0
                    reciprocal_ranks.append(0.0)
            score = (
                sum(reciprocal_ranks) / len(reciprocal_ranks)
                if reciprocal_ranks
                else 0.0
            )

        else:
            raise ValueError(f"Unknown scoring method: {method}")

        return {"score": score}

    def vis_skills(self, pred_skills, gt_skills, questions, savedir):
        """
        Visualize the skills for a given prediction and ground truth
        """
        """
        Create a nice skills_qualitative.txt at savedir/ that displays in two columns the pred_skills and gt_skills
        """
        output_file = os.path.join(savedir, "skills_qualitative.txt")
        # Create the directory if it doesn't exist
        os.makedirs(savedir, exist_ok=True)

        # Define column widths
        col_width = 25
        total_width = col_width * 3 + 5  # Extra space for padding

        def wrap_text(text, width):
            """Helper function to wrap text to specified width"""
            words = str(text).split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                # If a single word is longer than width, split it
                if len(word) > width:
                    while word:
                        if current_line:
                            lines.append(" ".join(current_line))
                            current_line = []
                            current_length = 0
                        lines.append(word[:width])
                        word = word[width:]
                    continue

                # Check if adding this word would exceed width
                if current_length + len(word) + (1 if current_line else 0) <= width:
                    current_line.append(word)
                    current_length += len(word) + (1 if current_line else 0)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(" ".join(current_line))
            return lines

        with open(output_file, "w") as f:
            # Write header with decorative borders
            f.write("=" * total_width + "\n")
            f.write("🔍 SKILLS QUALITATIVE COMPARISON REPORT 🔍\n")
            f.write("=" * total_width + "\n\n")

            # Write column headers
            f.write(
                f"{'QUESTIONS':<{40}}{'PREDICTED SKILLS':<{30}}{'GROUND TRUTH SKILLS':<{20}}\n"
            )
            f.write("-" * total_width + "\n")

            # Write content with wrapping
            for question, pred_skill, gt_skill in zip(
                questions, pred_skills, gt_skills
            ):
                # Wrap each column's content
                q_lines = wrap_text(question["question"], 40 - 2)
                pred_str = ", ".join(pred_skill)
                p_lines = wrap_text(pred_str, 30 - 2)
                gt_lines = wrap_text(gt_skill, 20 - 2)

                # Get the maximum number of lines needed
                max_lines = max(len(q_lines), len(p_lines), len(gt_lines))

                # Pad arrays to have equal length
                q_lines += [""] * (max_lines - len(q_lines))
                p_lines += [""] * (max_lines - len(p_lines))
                gt_lines += [""] * (max_lines - len(gt_lines))

                # Write each line
                for q, p, gt in zip(q_lines, p_lines, gt_lines):
                    f.write(f"{q:<{40}}{p:<{30}}{gt:<{20}}\n")

                # Add separator between entries
                f.write("-" * total_width + "\n")

    def vis_insights(self, pred_insights, gt_insights, data_dict, savedir):
        """
        Visualize the insights for a given prediction and ground truth
        Args:
            pred_insights (dict): Dictionary containing predictions
            gt_insights (dict): Dictionary containing ground truth
            data_dict (dict): Dictionary containing data and metadata
            savedir (str): Directory to save the visualization
        """
        output_file = os.path.join(savedir, "insights_comparison.txt")
        # Create the directory if it doesn't exist
        os.makedirs(savedir, exist_ok=True)
        with open(output_file, "w") as f:
            # Write header with decorative borders
            f.write("=" * 80 + "\n")
            f.write("🔍 INSIGHTS COMPARISON REPORT 🔍\n")
            f.write("=" * 80 + "\n\n")

            # Write questions from data_dict
            f.write("❓ QUESTIONS\n")
            f.write("-" * 40 + "\n")
            if "questions" in data_dict:
                for question in data_dict["questions"]:
                    f.write(f"• {question}\n")
            f.write("\n")

            # Write metadata
            f.write("📌 METADATA\n")
            f.write("-" * 40 + "\n")
            if "meta" in data_dict:
                for key, value in data_dict["meta"].items():
                    f.write(f"• {key}: {value}\n")
            f.write("\n")

            # Write predictions
            f.write("📊 PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            if isinstance(pred_insights, dict):
                for key, value in pred_insights.items():
                    f.write(f"• {key}:\n")
                    f.write(f"  {value}\n\n")
            else:
                f.write(f"• {pred_insights}\n\n")

            # Write ground truth
            f.write("\n📋 GROUND TRUTH\n")
            f.write("-" * 40 + "\n")
            if isinstance(gt_insights, dict):
                for key, value in gt_insights.items():
                    f.write(f"• {key}:\n")
                    f.write(f"  {value}\n\n")
            else:
                f.write(f"• {gt_insights}\n\n")

            f.write("=" * 80 + "\n")
