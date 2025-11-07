"""
Main script with chatbot-like interface using AgentGraph.
Allows interactive conversation with the data science agent.
"""

from agent_graph import AgentGraph
from data_handler import DataLoader
from transformers import pipeline
#from langchain_huggingface import HuggingFacePipeline
#from huggingface_hub import login
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

import os
import sys

# Load environment variables
load_dotenv()

def initialize_system():
    # """Initialize LLM, datasets, and agent graph"""

    # print("\n Logging in to HuggingFace...")
    # try:
    #     hf_token = os.getenv("HUGGINGFACE_TOKEN")
    #     if not hf_token:
    #         raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    #     login(token=hf_token, new_session=False)
    #     print(" Logged in successfully")
    # except Exception as e:
    #     print(f" Login failed: {e}")
    #     sys.exit(1)

    # print("\n Loading LLM ...")
    # try:
    #     pipe = pipeline(
    #         "text-generation",
    #         model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #         temperature=0.1,
    #         return_full_text=False
    #     )

    #     llm = HuggingFacePipeline(
    #         pipeline=pipe,
    #         model_kwargs={"temperature": 0.1}
    #     )
    #     print(" LLM loaded successfully")
    # except Exception as e:
    #     print(f" Error loading LLM: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)
    print("\nLoading Bedrock LLM...")
    try:
        # Instantiate ChatBedrock
        # It automatically picks up credentials and region from .env
        llm = ChatBedrock(
            # Specify the model ID for Llama 3.1 8B Instruct on Bedrock
            model_id="meta.llama3-1-8b-instruct-v1:0", 
            # Pass model parameters here
            model_kwargs={
                "temperature": 0.1,
                "max_gen_len": 4000 # Example if you want to set max tokens
            }
        )
        print("✓ Bedrock LLM (Llama 3.1 8B) loaded successfully")

    except Exception as e:
        print(f"✗ Error loading Bedrock LLM: {e}")
        import traceback
        traceback.print_exc()
        return
    print("\n Loading datasets...")
    loader = DataLoader()
    datasets = {}

    try:
        housing_path = "data/housing.csv"
        datasets['housing'] = loader.load_data(housing_path)
        print(f" Loaded housing dataset: {len(datasets['housing'])} rows, {len(datasets['housing'].columns)} columns")
    except Exception as e:
        print(f" Warning: Could not load housing dataset: {e}")

    try:
        coffee_path = "data/coffee_shop_sales.xlsx"
        datasets['coffee'] = loader.load_data(coffee_path)
        print(f" Loaded coffee dataset: {len(datasets['coffee'])} rows, {len(datasets['coffee'].columns)} columns")
    except Exception as e:
        print(f" Warning: Could not load coffee dataset: {e}")

    print("\n Initializing AgentGraph...")
    try:
        agent_graph = AgentGraph(llm=llm, datasets=datasets)
        print(" AgentGraph initialized successfully \n ")
        return agent_graph
    except Exception as e:
        print(f" Error initializing AgentGraph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_user_query(agent_graph: AgentGraph, user_input: str):
    """Process a user query and display results in chatbot style"""

    try:
        final_result = agent_graph.run(user_input)

        print("\n" + "=" * 80)
        print(" " * 30 + "FINAL REPORT")
        print("=" * 80 + "\n")

        print(final_result['response'])
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n Error processing query: {e}")
        import traceback
        traceback.print_exc()


def chat_loop(agent_graph: AgentGraph):
    """Main chatbot interaction loop"""

    while True:
        try:
            user_input = input("How may I help you? ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("\n Thank you for using the Data Science Agent. Goodbye!\n")
                break

            print()
            process_user_query(agent_graph, user_input)

        except KeyboardInterrupt:
            print("\n\n Session interrupted. Goodbye!\n")
            break
        except EOFError:
            print("\n\n Session ended. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print("\n You can continue asking questions or type 'exit' to exit.\n")


def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "DATA SCIENCE AGENT CHATBOT")
    print("=" * 80)
    print("Ask questions about your data and get comprehensive analysis!")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 80 + "\n")

    agent_graph = initialize_system()


    chat_loop(agent_graph)


if __name__ == "__main__":
    main()

