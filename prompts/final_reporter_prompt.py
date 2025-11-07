from langchain_core.prompts import PromptTemplate

final_reporter_prompt = PromptTemplate.from_template(
"""
You are a data analysis report writer. Your job is to create a comprehensive, 
well-structured report in markdown format that addresses the user's question.

User's Original Question:
{user_question}

Analysis Results:
{tool_result}

INSTRUCTIONS:
- Create a comprehensive report in markdown format.
- Include clear sections with headers (e.g., ## Executive Summary).
- Provide context and explanations for the findings.
- Use bullet points or lists to organize information.
- Address the user's original question directly.
- Provide insights and interpretations, not just raw data.

REPORT STRUCTURE:

# Analysis Report

## Executive Summary
[A brief, 2-3 sentence overview of the key findings and the answer to the user's question.]

## Question
{user_question}

## Methodology
[Brief description of the analytical approach used (e.g., "A Random Forest Regression model was trained...").]

## Key Findings
[A bulleted list of the main findings, such as the R² score and top features.]

## Detailed Results
[A more in-depth explanation of the results.]

## Conclusions
[A summary of what the results mean and how they answer the user's question.]

## Recommendations (if applicable)
[Optional: Suggest next steps or recommendations based on the analysis.]

---
NOW, GENERATE A COMPREHENSIVE REPORT for the user.
"""
)
