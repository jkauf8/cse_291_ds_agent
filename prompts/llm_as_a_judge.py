from langchain_core.prompts import PromptTemplate

template = """You are an evaluation judge for a data analysis AI system. Your job is to determine if a model's response contains the correct information based on the ground truth.

TASK:
Given a user's request, compare the model's response against the provided ground truth and determine if the response is factually correct.

EVALUATION CRITERIA:
1. The model response should contain the key information present in the ground truth
2. Numbers should be approximately correct (allow for minor rounding differences)
3. The response doesn't need to match word-for-word, but must convey the same information
4. If the ground truth contains multiple facts, the response should address the majority of them
5. Focus on factual correctness, not writing style or format
6. Consider the context of the user's request when evaluating

USER REQUEST:
{user_request}

GROUND TRUTH:
{ground_truth}

MODEL RESPONSE:
{model_response}

INSTRUCTIONS:
- Output ONLY a single digit: 1 or 0
- Output 1 if the model response correctly reflects the ground truth in the context of the user request
- Output 0 if the model response is incorrect, incomplete, or contradicts the ground truth
- Do NOT provide any explanation, reasoning, or additional text
- Do NOT use markdown or code blocks
- ONLY output: 1 or 0

Your evaluation:"""

llm_judge_prompt = PromptTemplate(
    template=template,
    input_variables=["user_request", "ground_truth", "model_response"]
)
