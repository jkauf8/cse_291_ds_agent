import os
from typing import Any

# from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser



from prompts.final_reporter_prompt import final_reporter_prompt

class FinalReporter:
    """
    LangChain agent that generates comprehensive, structured reports
    in markdown format based on the user's question and analysis results.
    """

    def __init__(self, llm):
        """
        Initialize the final reporter agent.

        Args:
            llm: An initialized LLM instance (e.g., ChatGoogleGenerativeAI)
        """
        self.llm = llm


    def generate_report(self, user_question: str, tool_result: Any):
        """
        Generate a comprehensive markdown report.

        Args:
            user_question: The original user's question
            tool_result: The result/output from the tool that was executed

        Returns:
            str: A comprehensive markdown-formatted report
        """
        reporter_agent = final_reporter_prompt | self.llm
        result = reporter_agent.invoke({
            "user_question": user_question,
            "tool_result": tool_result
        })

        # Extract text from response (handle both Gemini string and Bedrock AIMessage)
        if hasattr(result, 'content'):
            # Bedrock/LangChain returns AIMessage object with .content attribute
            return result.content
        else:
            # Gemini wrapper returns string directly
            return str(result)
