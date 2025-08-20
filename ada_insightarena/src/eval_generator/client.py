from openai import OpenAI
import base64
import re
from io import BytesIO
import json


class GPT4oClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def generate_questions(self, cell_batch, dataset_summary, skills, tasks):
        # Generate questions using GPT-4o
        prompt = self.create_prompt(cell_batch, dataset_summary, skills, tasks)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled data analytics assistant that generates structured questions and answers based on the provided information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200
        )
        return self.parse_response(response)

    def create_prompt(self, cell_batch, dataset_summary, skills, tasks):
        # Create the prompt for GPT-4o
        cell_details = []
        for cell_number, cell in cell_batch:
            cell_text = f"Cell {cell_number}:\n{cell.source}"
            if cell.cell_type == 'code' and cell.outputs:
                for output in cell.outputs:
                    if 'image_object' in output:
                        # Convert image to base64 string
                        image = output['image_object']
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        cell_text += f"\nOutput Image (base64): {img_str}"
            cell_details.append(cell_text)
        
        # Combine cell details into a structured format
        cell_text = "\n\n".join(cell_details)

        # Construct the skills and tasks section
        skills_section = "### Skills\nRelevant skills for performing analytics on this dataset are associated with specific tasks:\n\n"
        for task, related_skills in skills.items():
            skills_section += f"- **{task}**:\n  - " + "\n  - ".join(related_skills) + "\n"
        
        # Create the structured prompt
        prompt = (
            f"You have the following dataset: \n\n"
            f"### Dataset Overview\n"
            f"{dataset_summary}\n\n"
            f"### Notebook Cells\n"
            f"The following are the notebook cells provided to give context and examples of possible data analytics tasks:\n"
            f"{cell_text}\n\n"
            f"### Tasks\n"
            f"Relevant data analytics tasks include:\n"
            f"- " + "\n- ".join(tasks) + "\n\n"
            f"{skills_section}\n\n"
            f"- " + "\n- ".join(skills) + "\n\n"
            f"### Instructions\n"
            f"1. Generate a list of **questions and answers** related to **data analytics tasks** that can be performed on the dataset.\n"
            f"2. Each question should:\n"
            f"   - Focus on analyzing or gaining insights from the dataset itself (not the notebook).\n"
            f"   - Be framed from the perspective of someone analyzing the dataset directly.\n"
            f"   - Include the specific data analytics task and skill required to answer it.\n"
            f"3. Use the notebook cells as inspiration for possible types of analytics, but do not ask questions directly about the notebook's implementation.\n"
            f"4. For each generated question and answer, include:\n"
            f"   - The cell numbers that informed the question (if any).\n"
            f"   - The data analytics task and skill required.\n"
            f"5. Your answers to the question should be only come from the cells (usually the output cells or the markdown cells. Your answer should not be out of the given cell context.)\n"
            f"6. First choose the task, and the choose the skill needed to answer the question based on the list of skill of that specific task.\n\n"
            f"### Expected Output [IMPORTANT]\n"
            f"1. The question should be about data. Meaning, if a person sees the data, what analyitical question it might ask. The cells given from the notebook are only giving idea about type of analytics that can be done.\n"
            f"2. The answer should be derived from the cells (usually outputs). No analysis should be done outside the give cells. The cells are the only source of information for questions and answers.\n"
            f"3. Include different questions types, from basic data analysis questions for understanding the data to the detailed questions like asking about the number of clusters in the data which comes from doing a clustering (this has been done in the notebook.)\n"
            f"4. The task and skill should be selected from the list of tasks and skills provided.\n\n"
            f"### Output Format\n"
            f"Return a list of objects, where each object has the following structure:\n"
            f"- `question`: The generated question.\n"
            f"- `answer`: The corresponding answer.\n"
            f"- `task`: The data analytics task associated with the question.\n"
            f"- `skill`: The skill required to perform the task.\n"
            f"- `cell_ids`: The list of cell numbers used to inspire this question.\n\n"
            f"### Example Output\n"
            f"[\n"
            f"  {{\n"
            f"    \"question\": \"What is the distribution of sales across different regions?\",\n"
            f"    \"answer\": \"The distribution of sales shows that Region A accounts for 40% of total sales, Region B for 35%, and Region C for 25%.\",\n"
            f"    \"task\": \"Data distribution analysis\",\n"
            f"    \"skill\": \"Data visualization\",\n"
            f"    \"cell_ids\": [\"1\", \"2\", \"3\"]\n"
            f"  }},\n"
            f"  ...\n"
            f"]"
            f"In your response, just give the json file. No code or explanation is needed."
        )
        return prompt

    def parse_response(self, response):
        # Assuming response.text directly contains the JSON response
        response_text = response.choices[0].message.content
        try:
            # Directly parse the JSON from the response if well-formed
            parsed_json = json.loads(response_text)
            return parsed_json
        except json.JSONDecodeError as e:
            print("Failed to decode JSON from response:", str(e))
            # If the response is not a valid JSON, try regex extraction
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON from regex match:", str(e))
        return None

    def select_best_questions(self, questions):
        # Select the 10 best questions using GPT-4o
        prompt = (
            f"Select the 10 best questions from the following list.\n"
            f"The questions should be diverse and cover different aspects of\n"
            f"data analytics of the dataset (select them with different skills and tasks):\n\n"
            f"{questions}"
            f"### Output Format\n"
            f"Return a list of objects, where each object has the following structure:\n"
            f"- `question`: The generated question.\n"
            f"- `answer`: The corresponding answer.\n"
            f"- `task`: The data analytics task associated with the question.\n"
            f"- `skill`: The skill required to perform the task.\n"
            f"- `cell_numbers`: The list of cell numbers used to inspire this question (if applicable).\n\n"
            f"### Example Output\n"
            f"[\n"
            f"  {{\n"
            f"    \"question\": \"What is the distribution of sales across different regions?\",\n"
            f"    \"answer\": \"The distribution of sales shows that Region A accounts for 40% of total sales, Region B for 35%, and Region C for 25%.\",\n"
            f"    \"task\": \"Data distribution analysis\",\n"
            f"    \"skill\": \"Data visualization\"\n"
            f"  }},\n"
            f"  ...\n"
            f"]"
            f"In your response, just give the json file. No code or explanation is needed."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled data analytics assistant that generates structured questions and answers based on the provided information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200
        )
        return self.parse_response(response)
    
    def validate_answer(self, question, expected_answer, generated_answer):
        """Validates the generated answer against the expected answer using GPT."""
        try:
            prompt = (
                f"You are validating an answer from a model.\n"
                f"Question: {question}\n"
                f"Expected Answer: {expected_answer}\n"
                f"Generated Answer: {generated_answer}\n"
                f"Is the generated answer consistent with the expected answer? Provide a boolean response: True or False."
            )

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are natural language processing expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10
            )

            response_text = response.choices[0].message.content
            return "true" in response_text.lower()
        except Exception as e:
            print(f"Error in GPT validation: {e}")
            return False
        
    def generate_persona_goal(self, questions, dataset_summary):
        # Check if questions are provided
        if questions is None:
            print("No questions provided; skipping persona and goal generation.")
            return None  # Early exit if no questions are available
        prompt = (
            f"You are an expert data analyst who has just finished working with a dataset and the associated notebook content. "
            f"I will provide you with:\n\n"
            f"1. **A dataset summary**: This is a textual description of the dataset, including its columns, values, features, and overall purpose.\n"
            f"2. **Questions**: A list of dataset-specific questions reflecting insights derived from the dataset or the analysis described in the notebook.\n\n"
            f"Your task is to analyze these inputs and generate a JSON response containing:\n\n"
            f"- **Goal**: The primary goal of the analysis based on the dataset summary and notebook content. Describe what the analysis aims to achieve, without the models and analyses used as that should be what the analytics agent should figure out.\n"
            f"- **Persona**: A detailed description of the person conducting the analysis. Include information such as their profession, expertise level, goals, and interests. For example:\n"
            f"  *'A marketing analyst with 5 years of experience in e-commerce, focused on understanding customer behavior and optimizing marketing strategies for revenue growth.'*\n\n"
            f"### Instructions: \n\n"
            f"- The goal should be a one line and short description of what is the purpose of the analysis.\n"
            f"- The goal should be short and be \"what\" is the goal instead of \"how\" it is done.\n"
            f"- Goal is the the goal that a data analyst would have without telling him/her which methods to use.\n"
            f"- The persona should be detailed and should be a persona of a data analyst who is analyzing the data.\n"
            f"### Inputs:\n\n"
            f"- **Dataset Summary**: \"{dataset_summary}\"\n"
            f"- **Questions**: \"{json.dumps(questions, indent=4)}\"\n\n"
            f"### Output:\n\n"
            f"{{\n"
            f"  \"goal\": \"Describe the purpose of the analysis based on the inputs.\",\n"
            f"  \"persona\": \"Create a detailed persona of the individual conducting the analysis.\"\n"
            f"}}\n\n"
            f"Ensure your response is concise, well-structured, and grounded in the provided inputs. Generate the output as a valid JSON object."
            f"You should provide a only JSON file as the output. No additional information is needed."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled data analytics assistant that  analyzes different datasets using different skills."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100
        )
        return self.parse_response(response)
    