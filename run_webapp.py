"""
Main script with a Gradio web interface for the
AgentGraph chatbot.
"""

import pandas as pd
import gradio as gr
from agent_graph import AgentGraph
from gemini_llm import GeminiLLM
from langchain_aws import ChatBedrock
import os
import argparse
import sys
import traceback # Import traceback for better error handling
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_dataset(file_path: str):
    """Load a dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded housing dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return None


def initialize_system(use_bedrock: bool):
    """Initialize LLM, datasets, and agent graph"""

    # --- 1. Initialize LLM ---
    if use_bedrock:
        print("\nLoading AWS Bedrock LLM...")
        try:
            llm = ChatBedrock(
                model_id="meta.llama3-1-70b-instruct-v1:0",
                model_kwargs={
                    "temperature": 0.1,
                }
            )
            print("Bedrock LLM (Llama 3.1 70B) loaded successfully")
        except Exception as e:
            print(f"Error loading Bedrock LLM: {e}")
            traceback.print_exc()
            return None
    else:
        print("\nLoading Gemini LLM...")
        try:
            llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
            print("Gemini LLM (gemini-2.5-flash) loaded successfully")
        except Exception as e:
            print(f"Error loading Gemini LLM: {e}")
            traceback.print_exc()
            return None

    # --- 2. Load Datasets ---
    print("\nLoading datasets...")
    housing_df = load_dataset('data/housing.csv')
    if housing_df is None:
        return None

    datasets = {"housing": housing_df}

    # --- 3. Initialize AgentGraph ---
    print("\nInitializing AgentGraph...")
    try:
        agent_graph = AgentGraph(llm=llm, datasets=datasets)
        print("AgentGraph initialized successfully \n")
        return agent_graph
    except Exception as e:
        print(f"Error initializing AgentGraph: {e}")
        traceback.print_exc()
        return None


def create_gradio_app(agent_graph: AgentGraph, llm_name: str):
    """
    Creates and configures the Gradio ChatInterface.
    """

    def chat_response(message, history):
        """
        Core function for the Gradio ChatInterface.
        """
        print(f"\nProcessing user query: '{message}'")
        try:

            final_result = agent_graph.run(message)

            response = final_result.get('response', 'No response generated.')


            print("\n" + "=" * 80)
            print(" " * 30 + "FINAL REPORT (to console)")
            print("=" * 80 + "\n")
            print(response)
            print("\n" + "=" * 80 + "\n")

            # Return the response string to the Gradio UI
            return response

        except Exception as e:
            print(f"\nError processing query: {e}")
            traceback.print_exc()
            return "Sorry, I encountered an error while processing your request. Please check the console logs for details."

    # --- Create the Gradio Interface ---
    print("Creating Gradio interface...")
    title_llm = "Bedrock Llama 3.1 70B" if "Bedrock" in llm_name else "Gemini 2.5 Flash"

    iface = gr.ChatInterface(
        fn=chat_response,
        title=f"Data Science Agent ({title_llm})",
        description="Ask questions about the **housing dataset** and get a comprehensive analysis.",
        examples=[
            "Can you describe the housing dataset for me?",
            "Build a regression model to predict the price of houses.",
            "Can you predict the area of houses based on other features?",
            "First, describe the housing dataset, and then create a regression model to predict prices."
        ],
        theme="soft",
        submit_btn="Run Analysis",
    )

    return iface


if __name__ == "__main__":
    # Handle command-line argument for LLM choice
    parser = argparse.ArgumentParser(description="Data Science Agent with Housing Dataset")
    parser.add_argument(
        '--bedrock',
        action='store_true',
        help='Use AWS Bedrock Llama instead of Gemini (default)'
    )
    args = parser.parse_args()

    llm_name = "AWS Bedrock Llama" if args.bedrock else "Gemini"

    print("\n" + "=" * 80)
    print(f" " * 25 + f"DATA SCIENCE AGENT CHATBOT ({llm_name})")
    print("=" * 80)
    print("Initializing system... This may take a moment.")

    # 1. Initialize the agent graph ONCE.
    agent_graph = initialize_system(use_bedrock=args.bedrock)

    if agent_graph is None:
        print("=" * 80)
        print("FATAL: System initialization failed. Check logs above.")
        print("=" * 80)
        sys.exit(1) # Exit if setup fails

    print("=" * 80)
    print("System initialized. Launching Gradio interface...")
    print("=" * 80 + "\n")

    # 2. Create the Gradio App
    app = create_gradio_app(agent_graph, llm_name)

    # 3. Launch the web server
    app.launch()