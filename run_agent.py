"""
Interactive script to run the agent with custom queries.
"""

import pandas as pd
from agent_graph import AgentGraph
from gemini_llm import GeminiLLM
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def load_dataset(file_path: str):
    """Load a dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def save_result_to_file(question: str, response: str, filename: str = "query_results.md"):
    """Save a single query result to a markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n")
        f.write(f"## Query Time: {timestamp}\n\n")
        f.write(f"{response}\n\n")
    
    print(f"\nResult saved to {filename}\n")

def main():
    """Main function to run the interactive agent."""
    
    print("=" * 80)
    print(" " * 20 + "HOUSING DATA ANALYSIS AGENT")
    print("=" * 80)
    print("\nInitializing agent...")
    
    # Initialize the LLM
    llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    
    # Load the dataset
    housing_df = load_dataset('data/housing.csv')
    if housing_df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Loaded housing dataset: {len(housing_df)} rows, {len(housing_df.columns)} columns")

    datasets = {"housing": housing_df}

    # Initialize the agent graph
    agent = AgentGraph(llm=llm, datasets=datasets)
    print("Agent initialized successfully\n")
    
    print("=" * 80)
    print("\nYou can ask questions like:")
    print("  • 'Describe the housing dataset'")
    print("  • 'Build a regression model to predict house prices'")
    print("  • 'Predict the area of houses based on other features'")
    print("  • 'Create a model to predict the number of bedrooms'")
    print("  • 'What are the most important features for price prediction?'")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("=" * 80)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_query = input("\nYour question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n Thank you for using the Housing Data Analysis Agent. Goodbye!\n")
                break
            
            # Skip empty queries
            if not user_query:
                continue
            
            print("\n Processing your query...\n")
            print("-" * 80)
            
            # Run the agent
            result = agent.run(user_query)
            
            print("-" * 80)
            print("\n RESULT:\n")
            print(result.get('response', 'No response generated.'))
            print("\n" + "=" * 80)
            
            # Ask if user wants to save
            save_choice = input("\n Save this result to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_result_to_file(user_query, result.get('response', 'No response generated.'))
            
        except KeyboardInterrupt:
            print("\n\n Session interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n An error occurred: {e}")
            print("Please try again with a different question.\n")

if __name__ == "__main__":
    main()

