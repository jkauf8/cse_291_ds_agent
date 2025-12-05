import os

from langchain_core.output_parsers import JsonOutputParser



from prompts.reviewer_prompt import reviewer_prompt

class Reviewer:
    """
    LangChain agent that reviews the results from tools and decides whether
    the output is adequate or if the analysis needs to be re-planned.
    """

    def __init__(self, llm):
        """
        Initialize the reviewer agent.

        Args:
            llm: An initialized LLM instance (e.g., ChatGoogleGenerativeAI)
        """
        self.llm = llm


    def review(self, user_question: str, tool_result: str):
        """
        Execute the reviewer agent on a user question and tool result.

        Args:
            user_question: The original user's question
            tool_result: The result/output from the tool that was executed

        Returns:
            str: The routing decision ('planner' or 'final_reporter')
        """
        reviewer_agent = reviewer_prompt | self.llm | JsonOutputParser()
        result = reviewer_agent.invoke({
            "user_question": user_question,
            "tool_result": tool_result
        })
        
        return result["route"]
