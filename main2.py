"""
Main script with a Gradio web interface for the
AgentGraph chatbot.
"""

import gradio as gr
from agent_graph import AgentGraph
from data_handler import DataLoader
# from transformers import pipeline  
# from langchain_huggingface import HuggingFacePipeline
# from huggingface_hub import login 
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

import os
import sys
import traceback

# Load environment variables
load_dotenv()

def initialize_system():
    # """Initialize LLM, datasets, and agent graph"""

    print("\nLoading Bedrock LLM...")
    try:
        llm = ChatBedrock(
            model_id="meta.llama3-1-8b-instruct-v1:0", 
            model_kwargs={
                "temperature": 0.1,
            }
        )
        print("✓ Bedrock LLM (Llama 3.1 8B) loaded successfully")

    except Exception as e:
        print(f"✗ Error loading Bedrock LLM: {e}")
        traceback.print_exc()
        return None  
    
    print("\n Loading datasets...")
    loader = DataLoader()
    datasets = {}

    try:
        housing_path = "data/housing.csv"
        datasets['housing'] = loader.load_data(housing_path)
        print(f"✓ Loaded housing dataset: {len(datasets['housing'])} rows, {len(datasets['housing'].columns)} columns")
    except Exception as e:
        print(f"Warning: Could not load housing dataset: {e}")

    try:
        coffee_path = "data/coffee_shop_sales.xlsx"
        datasets['coffee'] = loader.load_data(coffee_path)
        print(f"✓ Loaded coffee dataset: {len(datasets['coffee'])} rows, {len(datasets['coffee'].columns)} columns")
    except Exception as e:
        print(f"Warning: Could not load coffee dataset: {e}")

    print("\n Initializing AgentGraph...")
    try:
        agent_graph = AgentGraph(llm=llm, datasets=datasets)
        print("✓ AgentGraph initialized successfully \n")
        return agent_graph
    except Exception as e:
        print(f"✗ Error initializing AgentGraph: {e}")
        traceback.print_exc()
        return None 

def create_gradio_app(agent_graph: AgentGraph):
    """
    Creates and configures the Gradio ChatInterface.
    """
    
    def chat_response(message, history):
        """
        This is the core function that Gradio's ChatInterface will call.
        'message' is the user's new input.
        'history' is the chat history (which we don't need for the agent).
        It uses the 'agent_graph' object initialized when the app started.
        """
        print(f"\nProcessing user query: '{message}'")
        try:
            
            final_result = agent_graph.run(message)
            
            response = final_result['response']
            
           
            print("\n" + "=" * 80)
            print(" " * 30 + "FINAL REPORT (to console)")
            print("=" * 80 + "\n")
            print(response)
            print("\n" + "=" * 80 + "\n")
            
            # Return the response string to the Gradio UI
            return response

        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            traceback.print_exc()
            # Return a user-friendly error message to the Gradio UI
            return "Sorry, I encountered an error while processing your request. Please check the console logs for details."

    # --- Create the Gradio Interface ---
    print("Creating Gradio interface...")
    iface = gr.ChatInterface(
        fn=chat_response,
        title="🤖 Data Science Agent",
        description="Ask questions about your data (housing.csv, coffee_shop_sales.xlsx) and get a comprehensive analysis.",
        examples=[
            "What's the average price of houses in the dataset?",
            "Which 5 neighborhoods have the highest average house price?",
            "What is the total coffee sales revenue?",
            "Generate a bar plot of sales by coffee type and save it as 'coffee_sales.png'."
        ],
        theme="soft",
        submit_btn="Run Analysis",
    )
    
    return iface


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "DATA SCIENCE AGENT CHATBOT")
    print("=" * 80)
    print("Initializing system... This may take a moment.")
    
    # 1. Initialize the agent graph ONCE.
    # This object will be shared by all Gradio sessions.
    agent_graph = initialize_system()

    if agent_graph is None:
        print("=" * 80)
        print("✗ FATAL: System initialization failed. Check logs above.")
        print("=" * 80)
        sys.exit(1) # Exit if setup fails

    print("=" * 80)
    print("✓ System initialized. Launching Gradio interface...")
    print("=" * 80 + "\n")

    # 2. Create the Gradio App
    app = create_gradio_app(agent_graph)
    
    # 3. Launch the web server
    # The 'chat_loop' is now replaced by Gradio's server
    app.launch()