import os

# from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser



from prompts.final_reporter_prompt import final_reporter_prompt

class FinalReporter:
    """
    LangChain agent that generates comprehensive, structured reports
    in markdown format based on the user's question and analysis results.
    """

    def __init__(self, llm=None, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the final reporter agent.

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


    def generate_report(self, user_question: str, tool_result: str):
        """
        Generate a comprehensive markdown report.

        Args:
            user_question: The original user's question
            tool_result: The result/output from the tool that was executed

        Returns:
            str: A comprehensive markdown-formatted report
        """
        reporter_agent = final_reporter_prompt | self.llm | StrOutputParser()
        result = reporter_agent.invoke({
            "user_question": user_question,
            "tool_result": tool_result
        })
        return result
