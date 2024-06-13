import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image
from cba.agents import Agent, AgentDemo
import os


@dataclasses.dataclass
class AgentState:
    """A class that keeps all conversation history and state."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    insights: List[str]
    questions: List[str]
    plots: List[str]
    csv: object = None
    gen_engine: str = "gpt-4-turbo-2024-04-09"
    output_root: str = f".tmp/outputs/"
    dataset_id: str = "1"
    current_answer: str = None
    current_analytics: str = None
    agent: AgentDemo = None
    version: str = "Unknown"
    skip_next: bool = False

    round = 0
    use_history: bool = True

    def add_plot(self, image):
        self.plots.append(image)
        if len(self.plots) > 3:
            self.plots.pop(0)

    def add_insight(self, insight):
        self.insights.append(insight)
        if len(self.insights) > 3:
            self.insights.pop(0)

    def add_question(self, questions):
        for question in questions:
            self.questions.append(question)
        if len(self.questions) > 3:
            self.questions = self.questions[-3:]

    def append_message(self, role, message):
        self.messages.append([role, message])

    def add_csv(self, df, path):
        self.csv = df
        self.csv_path = path

    def read_history(self, path, round=0):
        import json

        with open(path, "r") as f:
            history = json.load(f)

        questions = [
            history[i]["question"] for i in history if i.startswith(str(round))
        ]

        insights = [history[i]["answer"] for i in history if i.startswith(str(round))]

        plots = {}
        path_data = os.path.dirname(path)

        dirs_plots = [os.path.join(path_data, key) for key in history]
        for dir_path in dirs_plots:
            key = os.path.basename(dir_path)
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".jpg"):
                    image_path = os.path.join(dir_path, file_name)
                    plots[key] = image_path

        return insights, questions, plots

    def init_agent(self):
        # Get schema and goal
        goal = "I want to find interesting trends in this dataset"
        self.agent = AgentDemo(
            dataset_csv_path=self.csv_path,
            gen_engine=self.gen_engine,
            savedir=self.output_root,
            goal="I want to find interesting trends in this dataset",
            max_questions=3,
            branch_depth=4,
            n_retries=2,
        )

        return self.agent

    def to_gradio_chatbot(self):
        # We need to return a list of lists, where each inner list is a message from the agent and the user
        # In case there is no self.csv, we should keep asking for a CSV file
        ret = []
        for i, (role, msg) in enumerate(self.messages):
            if role == "Agent":
                # ret.append([None, msg])
                yield [None, msg]
            else:
                # ret.append([msg, None])
                yield [msg, None]
        return ret

    def copy(self):
        return AgentState(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            version=self.version,
            csv=self.csv,
            agent=self.agent,
            skip_next=self.skip_next,
            insights=self.insights,
            questions=self.questions,
            plots=self.plots,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "csv": self.csv,
            "version": self.version,
            "skip_next": self.skip_next,
            "insights": self.insights,
            "questions": self.questions,
            "plots": self.plots,
        }


cba = AgentState(
    system="",
    roles=("Agent", "User"),
    version="cba",
    messages=(
        (
            "Agent",
            "Hi! This is the Snow Agent, I'm here to help find insights in your data.",
        ),
        ("Agent", "Please upload a CSV file to get started."),
    ),
    csv=None,
    agent=None,
    skip_next=False,
    insights=["", "", ""],
    questions=["", "", ""],
    plots=[
        "docs/assets/white.png",
        "docs/assets/white.png",
        "docs/assets/white.png",
    ],
)
default_agent_state = cba
templates = {
    "default": default_agent_state,
}
