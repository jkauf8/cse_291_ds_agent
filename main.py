from data_handler import DataLoader
from agents.planner import Planner
import os
import sys
# from huggingface_hub import login
# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()


def main():

    # print("Logging in to HuggingFace...")
    # try:
    #     hf_token = os.getenv("HUGGINGFACE_TOKEN")
    #     if not hf_token:
    #         raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    #     login(token=hf_token, new_session=False)
    #     print("✓ Logged in successfully")
    # except Exception as e:
    #     print(f"✗ Login failed: {e}")

    # print("\nLoading HuggingFace LLM...")
    # try:
    #     print("Loading model pipeline...")

    #     pipe = pipeline(
    #         "text-generation",
    #         model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #         # max_new_tokens=100,
    #         temperature=0.1,
    #         return_full_text=False
    #     )

    #     llm = HuggingFacePipeline(
    #         pipeline=pipe,
    #         model_kwargs={"temperature": 0.1}
    #     )
    #     print("✓ LLM loaded successfully")

    # except Exception as e:
    #     print(f"✗ Error loading LLM: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return

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
                # "max_gen_len": 100 # Example if you want to set max tokens
            }
        )
        print("✓ Bedrock LLM (Llama 3.1 8B) loaded successfully")

    except Exception as e:
        print(f"✗ Error loading Bedrock LLM: {e}")
        import traceback
        traceback.print_exc()
        return


    print("=" * 80)
    print("Data Science Agent")
    print("=" * 80)

    loader = DataLoader()

    print("\n1. Loading datasets...")
    datasets = {}

    try:
        housing_path = "data/housing.csv"
        print(f"  Loading {housing_path}...")
        datasets['housing'] = loader.load_data(housing_path)
        print(f" Loaded {len(datasets['housing'])} rows, {len(datasets['housing'].columns)} columns")
    except Exception as e:
        print(f" Error loading Housing.csv: {e}")

    try:
        coffee_path = "data/coffee_shop_sales.xlsx"
        print(f"  Loading {coffee_path}...")
        datasets['coffee'] = loader.load_data(coffee_path)
        print(f" Loaded {len(datasets['coffee'])} rows, {len(datasets['coffee'].columns)} columns")
    except Exception as e:
        print(f" Error loading Coffee Shop Sales.xlsx: {e}")



    print("\n2. Initializing Planner Agent...")
    try:
        planner = Planner(llm=llm)
        print("✓ Planner initialized with HuggingFace LLM")
        print(f"  Available tools: {list(planner.tools.keys())}")
    except Exception as e:
        print(f"✗ Error initializing planner: {e}")
        return

    print("\n3. Executing Planner Agent...")
    print("-" * 80)

    # Example user request
    user_request = "What is the mean price of a house based on the data?"
    print(f"User Request: {user_request}")
    print("-" * 80)

    try:
        tool_to_use = planner.run(user_request)
        print(f"\n Planner selected tool: {tool_to_use}")

        print("\n4. Results:")
        print("-" * 80)
        print(f"Recommended Tool: {tool_to_use}")
        print(f"Tool Description: {planner.tools.get(tool_to_use, 'Tool not found')}")
        print("-" * 80)

    except Exception as e:
        print(f"\n Error running planner: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Analysis Pipeline Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
