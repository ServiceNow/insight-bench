# ===========================
# (1) Recommend Questions Prompts
# ===========================
def get_question_prompt(method="basic"):
    if method == "basic":
        prompt_template = GET_QUESTIONS_TEMPLATE
        system_template = GET_QUESTIONS_SYSTEM_MESSAGE
    if method == "follow_up":
        prompt_template = FOLLOW_UP_TEMPLATE
        system_template = FOLLOW_UP_SYSTEM_MESSAGE
    if method == "follow_up_with_type":
        prompt_template = FOLLOW_UP_TYPE_TEMPLATE
        system_template = FOLLOW_UP_SYSTEM_MESSAGE

    return prompt_template, system_template


# ===========================
# (2) CODE Prompts
# ===========================


def get_code_prompt(method="basic"):
    if method == "basic":
        code_template = GENERATE_CODE_TEMPLATE
    if method == "single":
        code_template = GENERATE_CODE_SINGLE_TEMPLATE
    elif method == "multi":
        code_template = GENERATE_CODE_TEMPLATE_MULTI

    return code_template


# ===========================
# (3) Interpret Prompt
# ===========================


def get_interpret_prompt(method):
    if method == "basic":
        prompt_template = INTERPRET_SOLUTION

    return prompt_template


# ===========================
# (4) Summarize Insights Prompt
# ===========================
def get_summarize_prompt(method="basic"):
    if method == "basic":
        prompt_template = SUMMARIZE_TEMPLATE
        system_template = SUMMARIZE_SYSTEM_MESSAGE

    return prompt_template, system_template


GET_QUESTIONS_TEMPLATE = """
### Instruction:

Given the following context:
<context>{context}</context>

Given the following goal:
<goal>{goal}</goal>

Given the following schema:
<schema>{schema}</schema>

Instructions:
* Write a list of questions to be solved by the data scientists in your team to explore my data and reach my goal.
* Explore diverse aspects of the data, and ask questions that are relevant to my goal.
* You must ask the right questions to surface anything interesting (trends, anomalies, etc.)
* Make sure these can realistically be answered based on the data schema.
* The insights that your team will extract will be used to generate a report.
* Each question should only have one part, that is a single '?' at the end which only require a single answer.
* Do not number the questions.
* You can produce at most {max_questions} questions. Stop generation after that.
* Most importantly, each question must be enclosed within <question></question> tags. Refer to the example response below:

Example response:
<question>What is the average age of the customers?</question>
<question>What is the distribution of the customers based on their age?</question>

### Response:
"""

GET_QUESTIONS_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""


INTERPRET_SOLUTION = """
### Instruction:
You are trying to answer a question based on information provided by a data scientist.

Given the context:
<context>
    You need to answer a question based on information provided by a data scientist.
</context>

Given the following dataset schema:
<schema>{schema}</schema>

Given the goal:
<goal>{goal}</goal>

Given the question:
<question>{question}</question>

Given the analysis:
<analysis>
    <message>
        {message}
    </message>
    {insights}
</analysis>

Instructions:
* Based on the analysis and other information provided above, write an answer to the question enclosed with <question></question> tags.
* The answer should be a single sentence, but it should not be too high level and should include the key details from justification.
* Write your answer in HTML-like tags, enclosing the answer between <answer></answer> tags, followed by a justification between <justification></justification> tags, followed by an insight between <insight></insight> tags.
* Refer to the following example response for the format of the answer and justification.
* The insight should be something interesting and grounded based on the question, goal, and the dataset schema, something that would be interesting. 
* The insight should be as quantiative as possible and informative and non-trivial and concise.
* The insight should be a meaningful conclusion that can be acquired from the analysis in laymans terms

Example response:
<answer>This is a sample answer</answer>
<insight>This is a sample insight</insight>
<justification>This is a sample justification</justification>

### Response:
"""


RETRY_TEMPLATE = """You failed.

Instructions:
-------------
{initial_prompt}
-------------

Completion:
-------------
{prev_output}
-------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
-------------
{error}
-------------

Please try again. Do not apologize. Please only respond with an answer that satisfies the constraints laid out in the Instructions:

"""


GET_INSIGHTS_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<schema>{schema}</schema>

Instructions:
* Produce a list of possible insights that we should look into to explore my data and reach my goal.
* Explore diverse aspects of the data, and present possible interesting insights (with explanation) that are relevant to my goal.
* Make sure these can realistically be based on the data schema.
* The insights that your team will extract will be used to insight a report.
* Each question that you produce must be enclosed in <insight></question> tags.
* Do not number the questions.
* You can produce at most {max_questions} insight.

"""

GET_INSIGHTS_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""


GET_DATASET_DESCRIPTION_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<schema>{schema}</schema>

Instructions:
* Generate a description of the dataset provided in the schema.
* The description should include the number of rows, columns, and a brief summary of the data.
* The description should be enclosed inside <description>content</description> tags.

"""

GET_DATASET_DESCRIPTION_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""

FOLLOW_UP_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<schema>{schema}</schema>

<question>{question}</question>

<answer>{answer}</answer>

Instructions:
* Produce a list of follow up questions explore my data and reach my goal.
* Note that we have already answered <question> and have the answer at <answer>, do not include a question similar to the one above. 
* Explore diverse aspects of the data, and ask questions that are relevant to my goal.
* You must ask the right questions to surface anything interesting (trends, anomalies, etc.)
* Make sure these can realistically be answered based on the data schema.
* The insights that your team will extract will be used to generate a report.
* Each question that you produce must be enclosed in <question>content</question> tags.
* Each question should only have one part, that is a single '?' at the end which only require a single answer.
* Do not number the questions.
* You can produce at most {max_questions} questions.

"""

FOLLOW_UP_TYPE_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<schema>{schema}</schema>

<question_type>{question_type}</question_type>

<question>{question}</question>

<answer>{answer}</answer>

Instructions:
* Produce a list of follow up questions explore my data and reach my goal.
* Note that we have already answered <question> and have the answer at <answer>, do not include a question similar to the one above. 
* Explore diverse aspects of the data, and ask questions that are relevant to my goal.
* You must ask the right questions to surface anything interesting (trends, anomalies, etc.)
* Make sure these can realistically be answered based on the data schema.
* The insights that your team will extract will be used to generate a report.
* The question has to adhere to the type of question that is provided in the <question_type> tag
* The type of question is either descriptive, diagnostic, prescriptive, or predictive.
* Each question that you produce must be enclosed in <question>content</question> tags.
* Each question should only have one part, that is a single '?' at the end which only require a single answer.
* Do not number the questions.
* You can produce at most {max_questions} questions.

"""


FOLLOW_UP_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""

SELECT_A_QUESTION_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<prev_questions>{prev_questions_formatted}</prev_questions>

<followup_questions>{followup_questions_formatted}</followup_questions>

Instructions:
* Given a context and a goal, select one follow up question from the above list to explore after prev_question that will help me reach my goal.
* Do not select a question similar to the prev_questions above. 
* Output only the index of the question in your response inside <question_id></question_id> tag.
* The output questions id must be 0-indexed.
"""

SELECT_A_QUESTION_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""


GENERATE_CODE_TEMPLATE = """

Given the goal:\n
{goal}

Given the schema:\n
{schema}

Given the data path:\n
{database_path}

Given the list of predefined functions in insightbench.tools module and their example usage:\n\n
{function_docs}

Give me the python code required to answer this question "{question}" and put a comment on top of each variable.\n\n

Make a single code block for starting with ```python
Do not produce code blocks for languages other than Python.
Make simple plots and save them as jpg files.
Import insightbench.tools, pandas as pd, and numpy as np at the beginning and use the predefined functions above to make plots.
If you need to make multiple line/histogram plots, plot with the same x-axis data should be plotted together.
For every plot, save a stats json file that stores the data of the plot.
For every plot, save a x and y axis json file.
There can be at most 100 datapoints in the plot.
Round floating datapoints values to the 100th decimal place if necessary.
Each json file must have a "name", "description", and "value" field that describes the data.
If the content of the json file is getting too long, truncate the unnecessary parts.
Call the fix_fnames function in insightbench.tools at the end of your code.
End your code with ```.

Output code:\n
"""

GENERATE_CODE_TEMPLATE_MULTI = """

Given the goal:\n
{goal}

Given the schema of the first dataset:\n
{schema}

Given the data path of the first dataset:\n
{database_path}

Given the schema of the second dataset:\n
{user_schema}

Given the data path of the second dataset:\n
{user_database_path}

Given the list of predefined functions in insightbench.tools module and their example usage:\n\n
{function_docs}

Give me the python code required to answer this question "{question}" and put a comment on top of each variable.\n\n

Make a single code block for starting with ```python
Do not produce code blocks for languages other than Python.
Import insightbench.tools at the beginning. 
You must only use the predefined functions mentioned above to make the plot.
You must generate one single simple plot and save it as a jpg file.
For the plot, save a stats json file that stores the data of the plot.
For the plot, save a x_axis.json and y_axis.json file that stores 100 most important x and y axis data points of the plot, respectively.
Save each json file using the insightbench.save_json function
For the json file must have a "name", "description", and "value" field that describes the data.
If the content of the json file is getting too long, truncate the unnecessary parts until the number of characters is less than 10000

Call the fix_fnames function in insightbench.tools at the end of your code.
End your code with ```.

Output code:\n
"""

GENERATE_CODE_SINGLE_TEMPLATE = """

Given the goal:\n
{goal}

Given the schema:\n
{schema}

Given the data path:\n
{database_path}

Given the list of predefined functions in insightbench.tools module and their example usage:\n\n
{function_docs}

Give me the python code required to answer this question "{question}" and put a comment on top of each variable.\n\n

Make a single code block for starting with ```python
Do not produce code blocks for languages other than Python.
Import insightbench.tools at the beginning. 
You must only use the predefined functions mentioned above to make the plot.
You must generate one single simple plot and save it as a jpg file.
For the plot, save a stats json file that stores the data of the plot.
For the plot, save a x_axis.json and y_axis.json file that stores a maximum of 50 of the most important x and y axis data points of the plot, respectively.
Save each json file using the insightbench.save_json function
For the json file must have a "name", "description", and "value" field that describes the data.
The content of the json file should be less than 4500 characters 

Call the fix_fnames function in insightbench.tools at the end of your code.
End your code with ```.

Output code:\n
"""


def get_g_eval_prompt(method="basic"):
    if method == "basic":
        geval_template, system_template = (
            G_EVAL_BASIC_TEMPLATE,
            G_EVAL_BASIC_SYSTEM_MESSAGE,
        )
    if method == "binary":
        geval_template, system_template = (
            G_EVAL_BINARY_TEMPLATE,
            G_EVAL_BINARY_SYSTEM_MESSAGE,
        )

    return geval_template, system_template


G_EVAL_BASIC_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provided Answer:
{answer}

Ground Truth Answer:
{gt_answer}

Follow these instructions when writing your response:
* On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the same as ground truth answer.
* Your response should contain only the numerical rating. DONOT include anything else like the provided answer, the ground truth answer, or an explanation of your rating scale in your response.
* Wrap your numerical rating inside <rating></rating> tags.
* Check very carefully before answering.
* Follow the output format as shown in the example below:
Example response:
<rating>7</rating>

### Response:

"""

G_EVAL_BINARY_SYSTEM_MESSAGE = """You are a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the provided response matches the ground truth answer."""

G_EVAL_BASIC_SYSTEM_MESSAGE = """You are a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the response answers the question based on the ground truth answer."""


G_EVAL_BINARY_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provided answer:
{answer}

GT Answer:
{gt_answer}

On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the the same as ground truth answer. The response should contain only the numerical rating.\
    
Check very carefully before answering.

### Response:
"""

G_EVAL_SYSTEM_MESSAGE = """You are a a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the response answers the question based on the ground truth answer."""


G_EVAL_M2M_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Predicted Answers:
{pred_list}

Grouth Truth Answers:
{gt_list}

For each ground truth answer above, provide the index of the most appropriate predicted answer (1-indexed).
Each line must contain a single integer value denoting the id of the matched prediction.
If there is no appropriate prediction for a ground truth answer, write -1.
Check very carefully before answering.

### Response:
"""

G_EVAL_M2M_SYSTEM_MESSAGE = "You are a high school teacher evaluating student responses to some questions. Before scoring their answers, you need to first match each ground truth answer with the most appropriate answer provided by the student."

SUMMARIZE_TEMPLATE = """
Hi, I require the services of your team to help me reach my goal.

<context>{context}</context>

<goal>{goal}</goal>

<history>{history}</history>

Instructions:
* Given a context and a goal, and all the history of <question_i><answer_i> pairs from the above list generate the 3 top actionable insights.
* Make sure they don't offer actions and the summary should be more about highlights of the findings
* Output each insight within this tag <insight></insight>.
* Each insight should be a meaningful conclusion that can be acquired from the analysis in laymans terms and should be as quantiative as possible and should aggregate the findings.
"""

SUMMARIZE_SYSTEM_MESSAGE = """
You the manager of a data science team whose goal is to help stakeholders within your company extract actionable insights from their data.
You have access to a team of highly skilled data scientists that can answer complex questions about the data.
You call the shots and they do the work.
Your ultimate deliverable is a report that summarizes the findings and makes hypothesis for any trend or anomaly that was found.
"""
