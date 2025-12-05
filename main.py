import pandas as pd
from agent_graph import AgentGraph
from gemini_llm import GeminiLLM
from langchain_aws import ChatBedrock
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

def load_dataset(file_path: str):
    """Load a dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None


def interactive_mode(agent):
    """Run the agent in interactive mode, allowing user to input queries."""
    print("================================================================================")
    print("INTERACTIVE MODE")
    print("================================================================================")
    print("\nEnter your questions about the housing dataset.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Your question: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break

            if not user_input:
                print("Please enter a question.\n")
                continue

            print("\nProcessing your question...\n")
            result = agent.run(user_input)
            response = result.get('response', 'No response generated.')

            print("================================================================================")
            print("RESPONSE:")
            print("================================================================================")
            print(response)
            print("================================================================================")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError processing question: {e}\n")


def main():
    """Main function to run the data analysis agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Data Science Agent with Housing Dataset")
    parser.add_argument(
        '--interact',
        action='store_true',
        help='Run in interactive mode (ask your own questions)'
    )
    parser.add_argument(
        '--bedrock',
        action='store_true',
        help='Use AWS Bedrock Llama instead of Gemini'
    )
    args = parser.parse_args()

    if args.bedrock:
        print("Initializing AWS Bedrock LLM...")
        try:
            llm = ChatBedrock(
                model_id="meta.llama3-1-70b-instruct-v1:0",
                model_kwargs={
                    "temperature": 0.1,
                }
            )
            print("Bedrock LLM initialized successfully")
            print("Model: Llama 3.1 70B Instruct")
        except Exception as e:
            print(f"Error initializing Bedrock LLM: {e}")
            return
    else:
        print("Initializing Gemini LLM...")
        llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
        print("Gemini LLM initialized successfully")

    # Load the dataset
    print("Loading housing dataset...")
    housing_df = load_dataset('data/housing.csv')
    if housing_df is None:
        return

    datasets = {"housing": housing_df}

    # Initialize the agent graph
    print("Initializing AgentGraph...")
    agent = AgentGraph(llm=llm, datasets=datasets)
    print("System ready!\n")

    # Check if interactive mode is requested
    if args.interact:
        interactive_mode(agent)
    else:
        # Run predefined queries (batch mode)
        print("Running in batch mode with predefined queries...")
        print("(Use --interact flag for interactive mode)\n")

        # Define the queries
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

        print("\nAnalysis complete. Results saved to analysis_results.md")

if __name__ == "__main__":
    main()