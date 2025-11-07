import os

# from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser



from prompts.planner_prompt import planner_prompt

class Planner:
    """
    LangChain agent that analyzes user requests and leverages appropriate tools
    to conduct analysis for the Network Intrusion Detection System.
    """

    def __init__(self, llm=None, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the analysis planner agent.

        Args:
            llm: External LLM instance (e.g., HuggingFacePipeline). If None, will try to use Bedrock
            model_id: Bedrock model ID to use (only if llm is None)
        """
        # Use provided LLM or initialize Bedrock LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = None
            # self.llm = ChatBedrock(
            #     model_id=model_id,
            #     region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-1'),
            #     model_kwargs={
            #         "max_tokens": 2048,
            #         "temperature": 0.7,
            #         "top_p": 0.9
            #     }
            # )

        # Set up tools
        self.tools = {
            "describe_data()": "Use this tool when the user asks for data overview, summary statistics, dataset information, data exploration, or wants to understand what's in the data. Examples: 'describe the data', 'show me summary stats', 'what columns are available', 'explore the dataset'",
            "run_regression()": "this tool is used for running RandomForest model to conduct regression on the data to discover regression insights including predictions, most important factors, etc",
            "direct_response()": "this tool is used for greetings, small talk, or questions unrelated to housing or coffee shop data analysis"
        }


    def plan(self, user_request):
        """
        Execute the agent on a user request.

        Args:
            user_request: The user's request

        Returns: 
            dict: The tool to use and its parameters
        """
        planner_agent = planner_prompt | self.llm | JsonOutputParser()
        result = planner_agent.invoke({"input": user_request, "tools": self.tools})
        return result

    def run(self, user_request):
        """
        Kept for backward compatibility if needed, but plan() is preferred.
        """
        return self.plan(user_request)

