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