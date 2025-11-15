import os
import json
from langchain_core.output_parsers import JsonOutputParser



from prompts.planner_prompt import planner_prompt

def extract_json_from_string(s):
    """Extracts a JSON object from a string."""
    try:
        # Find the start of the JSON object
        start_index = s.find('{')
        if start_index == -1:
            return None

        # Find the end of the JSON object
        end_index = s.rfind('}')
        if end_index == -1:
            return None

        # Extract the JSON string
        json_str = s[start_index:end_index + 1]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None

class Planner:
    """
    LangChain agent that analyzes user requests and leverages appropriate tools
    to conduct analysis for the housing data.
    """

    def __init__(self, llm):
        """
        Initialize the analysis planner agent.

        Args:
            llm: An initialized LLM instance (e.g., ChatGoogleGenerativeAI)
        """
        self.llm = llm

        # Set up tools
        self.tools = {
            "describe_data()": "Use this tool when the user asks for data overview, summary statistics, dataset information, data exploration, or wants to understand what's in the data. Examples: 'describe the data', 'show me summary stats', 'what columns are available', 'explore the dataset'",
            "run_regression()": "this tool is used for running RandomForest model to conduct regression on the data to discover regression insights including predictions, most important factors, etc",
            "direct_response()": "this tool is used for greetings, small talk, or questions unrelated to housing data analysis"
        }


    def plan(self, user_request):
        """
        Execute the agent on a user request.

        Args:
            user_request: The user's request

        Returns: 
            dict: The tool to use and its parameters
        """
        # Truncate the user_request to avoid exceeding the model's context window
        max_length = 1024
        if len(user_request) > max_length:
            user_request = user_request[:max_length]
            
        planner_agent = planner_prompt | self.llm
        raw_result = planner_agent.invoke({"input": user_request, "tools": self.tools})
        
        # Manually parse the JSON from the raw string
        json_result = extract_json_from_string(raw_result)
        
        if json_result is None:
            # Fallback or error handling if JSON parsing fails
            return {"tools": ["direct_response()"]}
            
        return json_result

    def run(self, user_request):
        """
        Kept for backward compatibility if needed, but plan() is preferred.
        """
        return self.plan(user_request)

