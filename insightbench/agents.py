import json
import os
import tempfile
import copy
import pandas as pd

from insightbench.utils import agent_utils as au
from insightbench import prompts
from langchain.schema import HumanMessage, SystemMessage
from insightbench.utils.metrics_utils import score_insight
from insightbench import metrics
from PIL import Image


class Agent:

    def __init__(
        self,
        table=None,
        table_user=None,
        dataset_csv_path=None,
        user_dataset_csv_path=None,
        model_name="gpt-4o",
        goal="I want to find interesting trends in this dataset",
        max_questions=3,
        branch_depth=4,
        n_retries=2,
        savedir=None,
        temperature=0,
    ):
        self.goal = goal
        self.max_questions = max_questions

        self.temperature = temperature
        self.model_name = model_name
        self.n_retries = n_retries
        self.branch_depth = branch_depth

        if savedir is None:
            savedir = tempfile.mkdtemp()
        self.savedir = savedir

        self.agent_poirot = AgentPoirot(
            model_name=model_name,
            savedir=savedir,
            goal=goal,
            verbose=True,
            temperature=temperature,
            n_retries=n_retries,
        )
        if dataset_csv_path is not None or table is not None:
            self.agent_poirot.set_table(
                table=table,
                table_user=table_user,
                dataset_csv_path=dataset_csv_path,
                user_dataset_csv_path=user_dataset_csv_path,
            )

    def get_insights(
        self,
        dataset_csv_path=None,
        user_dataset_csv_path=None,
        table=None,
        table_user=None,
        return_summary=True,
    ) -> tuple:
        """
        run the agent to generate a sequence of questions and answers
        """
        self.agent_poirot.set_table(
            table=table,
            table_user=table_user,
            dataset_csv_path=dataset_csv_path,
            user_dataset_csv_path=user_dataset_csv_path,
        )

        # Prompt 2: Get Root Questions
        root_questions = self.agent_poirot.recommend_questions(
            prompt_method="basic", n_questions=self.max_questions
        )

        # Go through the root questions and generate insights
        for q in root_questions:
            question = q
            for i in range(self.branch_depth):
                if self.agent_poirot.table_user is None:
                    prompt_code_method = "single"
                else:
                    prompt_code_method = "multi"
                _, insight_dict = self.agent_poirot.answer_question(
                    question,
                    prompt_code_method=prompt_code_method,
                    prompt_interpret_method="basic",
                )

                next_questions = self.agent_poirot.recommend_questions(
                    n_questions=self.max_questions,
                    insights_history=[insight_dict],
                    # prompt_method="follow_up_with_type",
                    # question_type="descriptive",
                )
                question = next_questions[
                    self.agent_poirot.select_a_question(next_questions)
                ]

        self.agent_poirot.save_state_dict(
            os.path.join(self.savedir, "insights_history.json")
        )
        pred_insights = [o["insight"] for o in self.agent_poirot.insights_history]
        if return_summary:
            pred_summary = self.summarize(self.agent_poirot.insights_history)
            return pred_insights, pred_summary
        return self.agent_poirot.insights_history

    def load_checkpoint(self, savedir):
        self.agent_poirot.load_state_dict(
            os.path.join(savedir, "insights_history.json")
        )

    def summarize(self, pred_insights, method="list", prompt_summarize_method="basic"):
        return self.agent_poirot.summarize(
            pred_insights, method, prompt_summarize_method
        )

    def evaluate_agent_on_summary(
        self, gt_insights_dict, score_name, return_summary=False
    ):
        # Get Summary Evaluation
        pred_summary = self.agent_poirot.summarize()
        gt_summary = gt_insights_dict["flag"]
        score_summary = score_insight(pred_summary, gt_summary, score_name=score_name)

        if return_summary:
            summary_dict = {
                "score_summary": score_summary,
                "pred_summary": pred_summary,
                "gt_summary": gt_summary,
            }
            return score_summary, summary_dict
        return score_summary

    def evaluate_agent_on_notebook(self, gt_flags_dict, score_method="rouge1"):
        """
        Evaluate the agent's performance
        """
        # get groundtruth
        gt_insights = []
        for o in gt_flags_dict["insights"]:
            gt_insights += [o["insight_dict"]["insight"]]

        pred_insights = [o["answer"] for o in self.agent_poirot.insights_history]
        # compute score using score_method
        if score_method == "rouge1":
            return metrics.compute_rouge(pred_insights, gt_insights)
        elif score_method == "g_eval":
            return metrics.compute_g_eval_o2m(pred_insights, gt_insights)
        elif score_method == "llama3_eval":
            return metrics.compute_llama3_eval_o2m(pred_insights, gt_insights)


class AgentPoirot:
    def __init__(
        self,
        savedir=None,
        context="This is a dataset that could potentially consist of interesting insights",
        model_name="gpt-3.5-turbo-0613",
        goal="I want to find interesting trends in this dataset",
        verbose=False,
        temperature=0,
        n_retries=2
    ):
        self.goal = goal
        if savedir is None:
            savedir = tempfile.mkdtemp()
        self.savedir = savedir
        self.context = context

        self.model_name = model_name
        self.temperature = temperature

        self.insights_history = []
        self.verbose = verbose
        self.n_retries = n_retries

    def set_table(
        self,
        table=None,
        table_user=None,
        dataset_csv_path=None,
        user_dataset_csv_path=None,
    ):
        self.table = table
        self.table_user = table_user
        self.dataset_csv_path = dataset_csv_path
        self.user_dataset_csv_path = user_dataset_csv_path

        if table is not None:
            self.table = table
        if table_user is not None:
            self.table_user = table_user
        if dataset_csv_path is not None:
            self.dataset_csv_path = dataset_csv_path
            self.table = pd.read_csv(dataset_csv_path)
        if user_dataset_csv_path is not None:
            self.user_dataset_csv_path = user_dataset_csv_path
            self.table_user = pd.read_csv(user_dataset_csv_path)

        # schema
        self.schema = au.get_schema(self.table)
        if self.table_user is not None:
            self.user_schema = au.get_schema(self.table_user)
        else:
            self.user_schema = None

    def summarize(self, pred_insights, method="list", prompt_summarize_method="basic"):
        if method == "list":
            chat = au.get_chat_model(self.model_name, self.temperature)

            # Function to format the data
            def format_data(data):
                result = ""
                for i, item in enumerate(data):
                    question_tag = f"<question_{i}>{item['question']}</question_{i}>\n"
                    answer_tag = f"<answer_{i}>{item['answer']}</answer_{i}>\n\n"
                    result += f"{question_tag} {answer_tag}\n"
                return result

            # Format the data and print
            formatted_history = format_data(pred_insights)

            # summary = agent.summarize_insights(method="list")
            content_prompt, system_prompt = prompts.get_summarize_prompt(
                method=prompt_summarize_method
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=content_prompt.format(
                        context=self.context,
                        goal=self.goal,
                        history=formatted_history,
                    )
                ),
            ]

            def _validate_tasks(out):
                isights = au.extract_html_tags(out, ["insight"])

                # Check that there are insights generated
                if "insight" not in isights:
                    return (
                        out,
                        False,
                        f"Error: you did not generate insights within the <insight></insight> tags.",
                    )
                isights = isights["insight"]
                return (isights, out), True, ""

            insight_list, message = au.chat_and_retry(
                chat, messages, n_retry=3, parser=_validate_tasks
            )

            insights = "\n".join(insight_list)

        return insights

    def select_a_question(self, questions):
        """
        Select a question from the list of questions
        """
        return au.select_a_question(
            questions,
            self.context,
            self.goal,
            [o["question"] for o in self.insights_history],
            self.model_name,
            prompts.SELECT_A_QUESTION_TEMPLATE,
            prompts.SELECT_A_QUESTION_SYSTEM_MESSAGE,
        )

    def generate_notebook():
        pass

    def generate_report():
        pass

    def recommend_questions(
        self,
        n_questions=3,
        insights_history=None,
        prompt_method=None,
        question_type=None,
    ):
        """
        Suggest Next Best Questions
        """
        if self.verbose:
            print(f"Generating {n_questions} Questions using {self.model_name}...")

        if insights_history is None:

            # Generate Root Questions
            questions = au.get_questions(
                prompt_method=prompt_method,
                context=self.context,
                goal=self.goal,
                messages=[],
                schema=self.schema,
                max_questions=n_questions,
                model_name=self.model_name,
                temperature=self.temperature,
            )
        else:
            # Generate Follow Up Questions
            last_insight = insights_history[-1]
            questions = au.get_follow_up_questions(
                context=self.context,
                goal=self.goal,
                question=last_insight["question"],
                answer=last_insight["answer"],
                schema=self.schema,
                max_questions=n_questions,
                model_name=self.model_name,
                prompt_method=prompt_method,
                question_type=question_type,
                temperature=self.temperature,
            )
            if self.verbose:
                print(
                    "\nFollowing up on the last insight:\n---------------------------------"
                )
                print(f"Question: {last_insight['question']}\n")
                print(f"Answer: {last_insight['answer']}\n")

        if self.verbose:
            print("\nNext Best Questions:\n-------------------")
            for idx, question in enumerate(questions):
                print(f"{idx+1}. {question}")
            print()

        return questions

    def answer_question(
        self,
        question,
        n_retries=2,
        return_insight_dict=True,
        prompt_code_method="single",
        prompt_interpret_method="interpret",
    ):
        n_retries = self.n_retries
        if self.verbose:
            print(f"Generating Code...")
        # Prompt 3: Generate Code
        code_output_folder = os.path.join(
            self.savedir, f"question_{str(len(self.insights_history))}"
        )

        if self.verbose:
            print(f"Interpreting Solution...")
            print(f"Results saved at: {self.savedir}")

        # with au.SuppressOutput():
        solution = au.generate_code(
            schema=self.schema,
            user_schema=self.user_schema,
            goal=self.goal,
            question=question,
            database_path=os.path.abspath(self.dataset_csv_path),
            user_database_path=(
                os.path.abspath(self.user_dataset_csv_path)
                if self.user_dataset_csv_path is not None
                else None
            ),
            output_folder=code_output_folder,
            model_name=self.model_name,
            n_retries=n_retries,
            prompt_method=prompt_code_method,
            temperature=self.temperature,
        )

        # Prompt 4: Interpret Solution
        interpretation_dict = au.interpret_solution(
            solution=solution,
            model_name=self.model_name,
            schema=self.schema,
            n_retries=n_retries,
            prompt_method=prompt_interpret_method,
            temperature=self.temperature,
        )
        answer = interpretation_dict["interpretation"]["answer"]

        if self.verbose:
            print("\nSolution\n---------")
            print(f"Question: {question}\n")
            print(f"Answer: {answer}\n")
            print(
                f"Justification: {interpretation_dict['interpretation']['justification']}\n"
            )

        insight_dict = {
            "question": question,
            "answer": answer,
            "insight": interpretation_dict["interpretation"]["insight"],
            "justification": interpretation_dict["interpretation"]["justification"],
            "output_folder": code_output_folder,
        }

        # Save into the savedir
        with open(os.path.join(code_output_folder, "insight.json"), "w") as json_file:
            json.dump(insight_dict, json_file, indent=4, sort_keys=True)

        # add to insights
        self.insights_history += [insight_dict]

        insight_dict = copy.deepcopy(insight_dict)
        insight_dict.update(self.get_insight_objects(insight_dict))

        if return_insight_dict:
            return answer, insight_dict

        return answer["answer"]

    def get_insight_objects(self, insight_dict):
        """
        Get Insight Objects
        """
        if os.path.exists(os.path.join(insight_dict["output_folder"], "plot.jpg")):
            # get plot.jpg
            plot = Image.open(os.path.join(insight_dict["output_folder"], "plot.jpg"))
        else:
            plot = None

        if os.path.exists(os.path.join(insight_dict["output_folder"], "x_axis.jpg")):
            # get x_axis.json
            x_axis = json.load(
                open(os.path.join(insight_dict["output_folder"], "x_axis.json"), "r")
            )
        else:
            x_axis = None

        if os.path.exists(os.path.join(insight_dict["output_folder"], "y_axis.json")):
            # get y_axis.json
            y_axis = json.load(
                open(os.path.join(insight_dict["output_folder"], "y_axis.json"), "r")
            )
        else:
            y_axis = None

        if os.path.exists(os.path.join(insight_dict["output_folder"], "stat.json")):
            try:
                # get stat.json
                stat = json.load(
                    open(os.path.join(insight_dict["output_folder"], "stat.json"), "r")
                )
            except:
                stat = None
        else:
            stat = None

        # get code.py
        if os.path.exists(os.path.join(insight_dict["output_folder"], "code.py")):
            code = open(
                os.path.join(insight_dict["output_folder"], "code.py"), "r"
            ).read()
        else:
            code = None

        insight_object = {
            "plot": plot,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "stat": stat,
            "code": code,
        }
        return insight_object

    def save_state_dict(self, fname):
        with open(fname, "w") as f:
            json.dump(self.insights_history, f, indent=4)

    def load_state_dict(self, fname):
        with open(fname, "r") as f:
            self.insights_history = json.load(f)
