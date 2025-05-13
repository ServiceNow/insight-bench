import pandas as pd
import openai
from typing import List
import os, ast
from pathlib import Path
from src.utils import get_llm_response
import json
from datetime import datetime


def generate_analytics_questions(df: pd.DataFrame, num_questions: int = 2) -> List[str]:
    """
    Generate relevant quantitative data analytics questions for a given CSV dataset.

    Args:
        df (pd.DataFrame): The base dataset
        num_questions (int): Number of questions to generate (default: 5)

    Returns:
        List[str]: List of generated questions
    """
    # Get dataset information
    columns = df.columns.tolist()
    sample_data = df.head(3).to_string()
    data_types = df.dtypes.to_string()

    # Construct prompt for the LLM
    prompt = f"""Given a dataset with the following characteristics:

Columns: {columns}
Data Types: {data_types}
Sample Data:
{sample_data}

Generate {num_questions} specific, quantitative data analytics questions that could be answered using this dataset. 
Focus on questions related to the following tasks. Each question should be paired with the relevant task name from the following list:
- Basic data analysis: Summarize the main features of the dataset, including measures such as mean, median, mode, variance, and standard deviation.
- Sentiment Analysis: Analyze the emotional tone and opinion expressed in the text.
- A/B Testing: Compare two or more variants to determine which performs better for a specific objective.
- Forecasting: Estimate future values or trends based on historical patterns and statistical methods.
- Fraud Detection: Identify suspicious patterns and anomalies to prevent fraudulent activities.
- Recommendation Systems: Generate personalized suggestions based on user preferences and item features.
- Churn Analysis: Predict customer departure patterns for retention strategies.
- Customer Segmentation: Group customers to enable targeted marketing and personalized services.
- Network Analysis: Analyze relationships and interactions between entities to understand system structure and behavior.
- Association Rule Mining: Uncover connections between items or events by analyzing co-occurrence patterns.
- Dashboard Summary: Transform complex data into concise, actionable insights through interactive displays and key metrics.
- Predictive Maintenance: Forecast equipment failures using machine learning to optimize maintenance timing.
- Cohort Analysis: Track and compare behavior patterns of user groups with shared characteristics over time.
- Attribution Modeling: Determine the impact of different marketing touchpoints on customer conversions.
- Anomaly Detection: Identify unusual patterns or outliers in the data.
- Feature Importance Ranking: Rank features based on their influence in the dataset.
- Geospatial Analysis: Discover geographic patterns and spatial relationships from location-based data.
- Causality: Identify cause-and-effect relationships between variables, controlling for confounding factors.
- Logs Clustering: Group similar log messages to detect patterns and anomalies in system behavior.
- Time Series Decomposition: Decompose time-based data into trend, seasonal, and residual components for analysis.
- Principal Component Analysis: Reduce dimensionality while preserving key patterns in the data.
- Correlation Analysis: Measure the strength and direction of relationships between variables.
- Knowledge Base: Extract meaningful insights and relationships from organizational knowledge.
- Multi-table Search: Optimize complex queries across multiple database tables.
- Huge Table Analysis: Process massive datasets using distributed computing and memory-efficient algorithms.
- Topic Modeling: Discover hidden themes in document collections through statistical pattern analysis.
- Market Analysis: Examine market trends, consumer behavior, and competitive dynamics for business decisions.
- Data Imputation: Fill missing values in datasets using statistical or machine learning methods.

Format each question on a new line, and pair it with its corresponding task name, like this:
1. [Task Name] - Question
2. [Task Name] - Question
...
Starting from 1 and ending at {num_questions}."""

    # Generate questions using OpenAI's API
    try:
        questions_text = get_llm_response(prompt)
        print(questions_text)
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        questions = [q.lstrip("0123456789. ") for q in questions]
        structured_questions = []
        for question in questions:
            parts = question.split("] - ", 1)
            if len(parts) == 2:
                task_name = parts[0].strip("[")
                question_text = parts[1]
                structured_questions.append(
                    {
                        "task": task_name,
                        "question": question_text,
                    }
                )
        
        return structured_questions

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []


def generate_analytics_questions_with_goal_persona(
    df: pd.DataFrame, goal: str, persona: str, num_questions: int = 2
) -> List[str]:
    """
    Generate relevant quantitative data analytics questions for a given CSV dataset.

    Args:
        csv_path (str): Path to the CSV file
        goal (str): Goal for the questions
        persona (str): Persona for the questions
        num_questions (int): Number of questions to generate (default: 5)

    Returns:
        List[str]: List of generated questions
    """
    # Try different encodings to read the CSV file
    # encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    # for encoding in encodings:
    #     try:
    #         df = pd.read_csv(csv_path, encoding=encoding)
    #         break
    #     except UnicodeDecodeError:
    #         continue
    #     except Exception as e:
    #         print(f"Error reading CSV with {encoding} encoding: {str(e)}")
    #         continue
    # else:  # If no encoding worked
    #     print(f"Failed to read {csv_path} with any of the attempted encodings")
    #     return []

    # Get dataset information
    columns = df.columns.tolist()
    sample_data = df.head(3).to_string()
    data_types = df.dtypes.to_string()

#Version-2 (Final)
    # Construct prompt for the LLM
    prompt = f"""Given a dataset with the following characteristics:

Columns: {columns}
Data Types: {data_types}
Sample Data:
{sample_data}

Additionally, consider the following project goal and persona:
- Goal: {goal}
- Persona: {persona}

Generate {num_questions} specific, advanced, and diverse quantitative data analytics questions that could be answered using this dataset. Ensure that the questions:

1. **Pertain to the Goal and Persona**: Each question must directly relate to the provided goal and persona. Avoid generating questions that deviate from the context of the goal or persona.
2. **Are Diverse and Varied**: The questions should cover a wide range of aspects of the dataset, including but not limited to trends, relationships, anomalies, and actionable insights. Ensure no single area is overrepresented.
3. **Are Advanced**: The questions should require deeper analytical thinking, such as multivariate analysis, predictive modeling, or advanced statistical techniques. Avoid basic or superficial questions.

Each question should be paired with the relevant task name from the following list:

- Basic data analysis: Summarize the main features of the dataset, including measures such as mean, median, mode, variance, and standard deviation.
- Sentiment Analysis: Analyze the emotional tone and opinion expressed in the text.
- A/B Testing: Compare two or more variants to determine which performs better for a specific objective.
- Forecasting: Estimate future values or trends based on historical patterns and statistical methods.
- Fraud Detection: Identify suspicious patterns and anomalies to prevent fraudulent activities.
- Recommendation Systems: Generate personalized suggestions based on user preferences and item features.
- Churn Analysis: Predict customer departure patterns for retention strategies.
- Customer Segmentation: Group customers to enable targeted marketing and personalized services.
- Network Analysis: Analyze relationships and interactions between entities to understand system structure and behavior.
- Association Rule Mining: Uncover connections between items or events by analyzing co-occurrence patterns.
- Dashboard Summary: Transform complex data into concise, actionable insights through interactive displays and key metrics.
- Predictive Maintenance: Forecast equipment failures using machine learning to optimize maintenance timing.
- Cohort Analysis: Track and compare behavior patterns of user groups with shared characteristics over time.
- Attribution Modeling: Determine the impact of different marketing touchpoints on customer conversions.
- Anomaly Detection: Identify unusual patterns or outliers in the data.
- Feature Importance Ranking: Rank features based on their influence in the dataset.
- Geospatial Analysis: Discover geographic patterns and spatial relationships from location-based data.
- Causality: Identify cause-and-effect relationships between variables, controlling for confounding factors.
- Logs Clustering: Group similar log messages to detect patterns and anomalies in system behavior.
- Time Series Decomposition: Decompose time-based data into trend, seasonal, and residual components for analysis.
- Principal Component Analysis: Reduce dimensionality while preserving key patterns in the data.
- Correlation Analysis: Measure the strength and direction of relationships between variables.
- Knowledge Base: Extract meaningful insights and relationships from organizational knowledge.
- Multi-table Search: Optimize complex queries across multiple database tables.
- Huge Table Analysis: Process massive datasets using distributed computing and memory-efficient algorithms.
- Topic Modeling: Discover hidden themes in document collections through statistical pattern analysis.
- Market Analysis: Examine market trends, consumer behavior, and competitive dynamics for business decisions.
- Data Imputation: Fill missing values in datasets using statistical or machine learning methods.

Format each question on a new line, and pair it with its corresponding task name, like this:
1. [Task Name] - Question
2. [Task Name] - Question
...
Starting from 1 and ending at {num_questions}...

For example:
1. [Forecasting] - Using time series decomposition, predict the seasonal trends in customer engagement over the next 12 months, specifically focusing on how these trends align with the goal of increasing user retention for the persona of a subscription-based business.
2. [Anomaly Detection] - Identify unusual patterns in user behavior that may indicate fraudulent activity, and propose methods to mitigate these risks, ensuring the solutions align with the goal of reducing fraud for the persona of a financial services provider.
3. [Customer Segmentation] - Apply clustering algorithms to segment customers based on purchasing behavior and sentiment analysis, and recommend targeted marketing strategies for each segment, ensuring the recommendations align with the goal of increasing sales for the persona of an e-commerce platform.
4. [Causality] - Investigate the causal relationship between marketing spend and customer conversion rates, controlling for external factors such as seasonality and economic conditions, and provide insights that align with the goal of optimizing marketing ROI for the persona of a digital marketing agency.
5. [Feature Importance Ranking] - Rank the most influential features in predicting customer churn using SHAP values, and explain how these features impact retention strategies, ensuring the analysis aligns with the goal of reducing churn for the persona of a telecom company.

Ensure that the questions are advanced, diverse, and directly relevant to the goal and persona.
"""
#Version-1
    
#     prompt = f"""Given a dataset with the following characteristics:

# Columns: {columns}
# Data Types: {data_types}
# Sample Data:
# {sample_data}

# Additionally, consider the following project goal and persona:
# - Goal: {goal}
# - Persona: {persona}

# Generate {num_questions} specific, advanced, and diverse quantitative data analytics questions that could be answered using this dataset. Ensure that the questions cover a wide range of aspects and require deeper analytical thinking. Each question should be paired with the relevant task name from the following list:

# - Basic data analysis: Summarize the main features of the dataset, including measures such as mean, median, mode, variance, and standard deviation.
# - Sentiment Analysis: Analyze the emotional tone and opinion expressed in the text.
# - A/B Testing: Compare two or more variants to determine which performs better for a specific objective.
# - Forecasting: Estimate future values or trends based on historical patterns and statistical methods.
# - Fraud Detection: Identify suspicious patterns and anomalies to prevent fraudulent activities.
# - Recommendation Systems: Generate personalized suggestions based on user preferences and item features.
# - Churn Analysis: Predict customer departure patterns for retention strategies.
# - Customer Segmentation: Group customers to enable targeted marketing and personalized services.
# - Network Analysis: Analyze relationships and interactions between entities to understand system structure and behavior.
# - Association Rule Mining: Uncover connections between items or events by analyzing co-occurrence patterns.
# - Dashboard Summary: Transform complex data into concise, actionable insights through interactive displays and key metrics.
# - Predictive Maintenance: Forecast equipment failures using machine learning to optimize maintenance timing.
# - Cohort Analysis: Track and compare behavior patterns of user groups with shared characteristics over time.
# - Attribution Modeling: Determine the impact of different marketing touchpoints on customer conversions.
# - Anomaly Detection: Identify unusual patterns or outliers in the data.
# - Feature Importance Ranking: Rank features based on their influence in the dataset.
# - Geospatial Analysis: Discover geographic patterns and spatial relationships from location-based data.
# - Causality: Identify cause-and-effect relationships between variables, controlling for confounding factors.
# - Logs Clustering: Group similar log messages to detect patterns and anomalies in system behavior.
# - Time Series Decomposition: Decompose time-based data into trend, seasonal, and residual components for analysis.
# - Principal Component Analysis: Reduce dimensionality while preserving key patterns in the data.
# - Correlation Analysis: Measure the strength and direction of relationships between variables.
# - Knowledge Base: Extract meaningful insights and relationships from organizational knowledge.
# - Multi-table Search: Optimize complex queries across multiple database tables.
# - Huge Table Analysis: Process massive datasets using distributed computing and memory-efficient algorithms.
# - Topic Modeling: Discover hidden themes in document collections through statistical pattern analysis.
# - Market Analysis: Examine market trends, consumer behavior, and competitive dynamics for business decisions.
# - Data Imputation: Fill missing values in datasets using statistical or machine learning methods.

# Format each question on a new line, and pair it with its corresponding task name, like this:
# 1. [Task Name] - Question
# 2. [Task Name] - Question
# ...
# Starting from 1 and ending at {num_questions}.

# Ensure that the questions are:
# - **Advanced**: Require deeper analytical thinking, such as multivariate analysis, predictive modeling, or advanced statistical techniques.
# - **Diverse**: Cover a wide range of aspects of the dataset, including temporal trends, relationships between variables, anomalies, and actionable insights.
# - **Comprehensive**: Collectively provide a holistic understanding of the dataset, addressing both high-level trends and granular details.

# For example:
# 1. [Forecasting] - Using time series decomposition, predict the seasonal trends in customer engagement over the next 12 months and identify key drivers of peak activity.
# 2. [Anomaly Detection] - Identify unusual patterns in user behavior that may indicate fraudulent activity, and propose methods to mitigate these risks.
# 3. [Customer Segmentation] - Apply clustering algorithms to segment customers based on purchasing behavior and sentiment analysis, and recommend targeted marketing strategies for each segment.
# 4. [Causality] - Investigate the causal relationship between marketing spend and customer conversion rates, controlling for external factors such as seasonality and economic conditions.
# 5. [Feature Importance Ranking] - Rank the most influential features in predicting customer churn using SHAP values, and explain how these features impact retention strategies.

# Make sure the questions are not basic or superficial but instead require advanced analytical techniques and provide actionable insights.
# """

    # Generate questions using OpenAI's API
    try:
        questions_text = get_llm_response(prompt)
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        questions = [q.lstrip("0123456789. ") for q in questions]
        structured_questions = []
        for question in questions:
            parts = question.split("] - ", 1)
            if len(parts) == 2:
                task_name = parts[0].strip("[")
                question_text = parts[1]
                structured_questions.append(
                    {
                        "task": task_name,
                        "question": question_text,
                    }
                )
        
        return structured_questions

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []

def generate_analytics_questions_iterative_with_goal_persona(
    df: pd.DataFrame, goal: str, persona: str, num_questions: int = 2
) -> List[str]:
    columns = df.columns.tolist()
    sample_data = df.head(3).to_string()
    data_types = df.dtypes.to_string()
    data_analysis_ques_prompt= f'''You are an AI assistant specializing in data analysis.
    I have a dataset with the following details:

    Columns: {columns}
    Data Types: {data_types}
    Sample Data: {sample_data}
    Goal: {goal}
    Persona: {persona}

    Based on this information, generate five insightful questions that a data analyst in this persona would ask
    or seek to answer when exploring the dataset. The questions should be relevant to the dataset’s structure and align with the stated goal 
    of the analysis.

    Make sure that all the questions are returned as a list named generated_questions
    The generation format should be: generated_questions = [question_1, question_2, ..., question_5]
    '''
    try:
        generated_basic_questions = get_llm_response(data_analysis_ques_prompt)
    except Exception as e:
        print(f"Error generating basic questions: {str(e)}")
        return []

    advanced_ques_prompt = f'''You are an AI assistant specializing in data analysis. I have a dataset with the following details:

    Columns: {columns}
    Data Types: {data_types}
    Sample Data: {sample_data}
    Goal: {goal}
    Persona: {persona}
    Additionally, I have already generated these **basic questions** that a data analyst might ask when exploring this dataset:
    {generated_basic_questions}
    Now, using the provided dataset information, these basic questions, and the goal and persona as guiding principles, **generate {num_questions} additional advanced and diverse questions that require specialized analytical techniques** to answer.

    Requirements for the **advanced questions**:
    **Goal Alignment**: Each question must directly contribute to achieving the stated goal of the analysis.
    **Persona Relevance**: The complexity and focus of the questions should match the persona’s expertise and domain.
    **Higher Complexity**: Questions should **require** deeper analytical skills, making them significantly more advanced than the basic ones.
    **Skill-Based**: Each question should necessitate the use of exactly one skill from the following skill list:

        - Sentiment Analysis: Analyze the emotional tone and opinion expressed in the text.
        - A/B Testing: Compare two or more variants to determine which performs better for a specific objective.
        - Forecasting: Estimate future values or trends based on historical patterns and statistical methods.
        - Fraud Detection: Identify suspicious patterns and anomalies to prevent fraudulent activities.
        - Recommendation Systems: Generate personalized suggestions based on user preferences and item features.
        - Churn Analysis: Predict customer departure patterns for retention strategies.
        - Customer Segmentation: Group customers to enable targeted marketing and personalized services.
        - Network Analysis: Analyze relationships and interactions between entities to understand system structure and behavior.
        - Association Rule Mining: Uncover connections between items or events by analyzing co-occurrence patterns.
        - Dashboard Summary: Transform complex data into concise, actionable insights through interactive displays and key metrics.
        - Predictive Maintenance: Forecast equipment failures using machine learning to optimize maintenance timing.
        - Cohort Analysis: Track and compare behavior patterns of user groups with shared characteristics over time.
        - Attribution Modeling: Determine the impact of different marketing touchpoints on customer conversions.
        - Anomaly Detection: Identify unusual patterns or outliers in the data.
        - Feature Importance Ranking: Rank features based on their influence in the dataset.
        - Geospatial Analysis: Discover geographic patterns and spatial relationships from location-based data.
        - Causality: Identify cause-and-effect relationships between variables, controlling for confounding factors.
        - Logs Clustering: Group similar log messages to detect patterns and anomalies in system behavior.
        - Time Series Decomposition: Decompose time-based data into trend, seasonal, and residual components for analysis.
        - Principal Component Analysis: Reduce dimensionality while preserving key patterns in the data.
        - Correlation Analysis: Measure the strength and direction of relationships between variables.
        - Knowledge Base: Extract meaningful insights and relationships from organizational knowledge.
        - Multi-table Search: Optimize complex queries across multiple database tables.
        - Huge Table Analysis: Process massive datasets using distributed computing and memory-efficient algorithms.
        - Topic Modeling: Discover hidden themes in document collections through statistical pattern analysis.
        - Market Analysis: Examine market trends, consumer behavior, and competitive dynamics for business decisions.
        - Data Imputation: Fill missing values in datasets using statistical or machine learning methods.

    -**Implicit Skill Usage**: The skill name must not be directly mentioned in the question.
    -**Diverse Techniques**: Ensure a variety of skills are used across the five questions, avoiding redundancy.

    Before finalizing a question, **internally reason** if **GPT-4o can answer this question using basic reasoning or common-sense knowledge?**
    - If *yes*, reject the question and generate a more advanced one.
    - If *no*, proceed.

    Format each question on a new line, and pair it with its corresponding task name, like this:
    1. [Task Name] - Question
    2. [Task Name] - Question
    ...
    Starting from 1 and ending at {num_questions}...

    For example:
    1. [Forecasting] - Using time series decomposition, predict the seasonal trends in customer engagement over the next 12 months, specifically focusing on how these trends align with the goal of increasing user retention for the persona of a subscription-based business.
    2. [Anomaly Detection] - Identify unusual patterns in user behavior that may indicate fraudulent activity, and propose methods to mitigate these risks, ensuring the solutions align with the goal of reducing fraud for the persona of a financial services provider.
    3. [Customer Segmentation] - Apply clustering algorithms to segment customers based on purchasing behavior and sentiment analysis, and recommend targeted marketing strategies for each segment, ensuring the recommendations align with the goal of increasing sales for the persona of an e-commerce platform.
    4. [Causality] - Investigate the causal relationship between marketing spend and customer conversion rates, controlling for external factors such as seasonality and economic conditions, and provide insights that align with the goal of optimizing marketing ROI for the persona of a digital marketing agency.
    5. [Feature Importance Ranking] - Rank the most influential features in predicting customer churn using SHAP values, and explain how these features impact retention strategies, ensuring the analysis aligns with the goal of reducing churn for the persona of a telecom company.
    '''
    try:
        questions_text = get_llm_response(advanced_ques_prompt)
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        questions = [q.lstrip("0123456789. ") for q in questions]
        structured_questions = []
        for question in questions:
            parts = question.split("] - ", 1)
            if len(parts) == 2:
                task_name = parts[0].strip("[")
                question_text = parts[1]
                structured_questions.append(
                    {
                        "task": task_name,
                        "question": question_text,
                    }
                )
        
        return structured_questions

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []
# Example usage
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Initialize or load existing JSON file
    json_path = results_dir / "questions.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
    else:
        all_questions = {}

    # Get all CSV files and their corresponding goal.json files
    datasets_dir = Path("data/csvs")
    jsons_dir = Path("data/jsons")
    csv_files = list(datasets_dir.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/datasets directory!")
        exit()

    print(f"Found {len(csv_files)} CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'-'*50}")
        print(f"Processing file {i}/{len(csv_files)}: {csv_file}")
        print(f"{'-'*50}")

        # Skip if this file has already been processed
        if str(csv_file) in all_questions:
            print(f"Skipping {csv_file} - already processed")
            continue

        # Try to find corresponding goal.json file
        folder_num = csv_file.parent.name
        goal_json_path = jsons_dir / folder_num / "goal.json"

        if goal_json_path.exists():
            with open(goal_json_path, "r", encoding="utf-8") as f:
                goal_data = json.load(f)
                questions = generate_analytics_questions_with_goal_persona(
                    str(csv_file),
                    goal_data.get("goal", ""),
                    goal_data.get("persona", ""),
                )
        else:
            print(
                f"No goal.json found for {csv_file}, using default question generation"
            )
            questions = generate_analytics_questions(str(csv_file))

        # Create dataset entry with metadata first
        dataset_entry = {"dataset": str(csv_file)}

        # Add goal and persona first if available
        if goal_json_path.exists():
            dataset_entry["goal"] = goal_data.get("goal", "")
            dataset_entry["persona"] = goal_data.get("persona", "")

        # Add questions last
        dataset_entry["questions"] = []
        for question in questions:
            parts = question.split("] - ", 1)
            if len(parts) == 2:
                task_name = parts[0].strip("[")
                question_text = parts[1]
                question_data = {
                    "task_name": task_name,
                    "question": question_text,
                }
                dataset_entry["questions"].append(question_data)

        # Store and write questions for this dataset immediately
        all_questions[str(csv_file)] = dataset_entry
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)

        print(f"Saved questions for {csv_file}")

    print(f"\nAll results saved to: {json_path}")
