import os

# from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser



from prompts.reviewer_prompt import reviewer_prompt

class Reviewer:
    """
    LangChain agent that reviews the results from tools and decides whether
    the output is adequate or if the analysis needs to be re-planned.
    """

    def __init__(self, llm=None, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the reviewer agent.

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
