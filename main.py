"""
Main script with chatbot-like interface using AgentGraph.
Allows interactive conversation with the data science agent.
"""

import pandas as pd
from agent_graph import AgentGraph
from gemini_llm import GeminiLLM
import os
from dotenv import load_dotenv

load_dotenv()

def load_dataset(file_path: str):
    """Load a dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def main():
    """Main function to run the data analysis agent."""
    # Initialize the LLM
    llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

    # Load the dataset
    housing_df = load_dataset('data/housing.csv')
    if housing_df is None:
        return

    datasets = {"housing": housing_df}

    # Initialize the agent graph
    agent = AgentGraph(llm=llm, datasets=datasets)

    # Define the queries - testing different target variables
    queries = [
        {"type": "describe", "question": "Can you describe the housing dataset for me?"},
        {"type": "regression_price", "question": "Build a regression model to predict the price of houses."},
        {"type": "regression_area", "question": "Can you predict the area of houses based on other features?"},
        {"type": "regression_bedrooms", "question": "Build a regression model to predict the number of bedrooms."},
        {"type": "both", "question": "First, describe the housing dataset, and then create a regression model to predict prices."}
    ]

    # Run the queries and store the results
    results = []
    for query in queries:
        print(f"--- Running query ({query['type']}): {query['question']} ---")
        result = agent.run(query['question'])
        results.append({
            "query_type": query['type'],
            "question": query['question'],
            "response": result.get('response', 'No response generated.')
        })
        print(f"--- Finished query ({query['type']}) ---")

    # Save results to a markdown file
    with open("analysis_results.md", "w") as f:
        for result in results:
            f.write(f"# Query Type: {result['query_type']}\n\n")
            f.write(f"## Question:\n{result['question']}\n\n")
            f.write(f"## Answer:\n{result['response']}\n\n")
            f.write("---\n\n")

    print("Analysis complete. Results saved to analysis_results.md")

if __name__ == "__main__":
    main()