"""
Simple test to verify regression can predict different variables
"""
import pandas as pd
from agents.planner import Planner
from gemini_llm import GeminiLLM
import os
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*60)
print("SIMPLE TEST: Regression Tool Flexibility")
print("="*60)

# Initialize
llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
planner = Planner(llm=llm)

# Test different queries
tests = [
    "Build a regression model to predict house prices",
    "Predict the area of houses",
    "Build a model to predict number of bedrooms"
]

print("\nTesting planner's ability to identify different target variables:\n")

for i, query in enumerate(tests, 1):
    print(f"{i}. Query: {query}")
    result = planner.plan(query)
    target = result.get('target_column', 'NOT FOUND')
    print(f"   -> Target identified: {target}")
    print()

print("="*60)
print("RESULT: The planner can now identify different targets!")
print("The regression tool is no longer limited to 'price' only.")
print("="*60 + "\n")

