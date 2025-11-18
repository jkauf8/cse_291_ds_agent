"""
Baseline testing script for evaluating base LLM performance on describe_data questions.
Tests the raw LLM without any agent system to establish baseline performance.
"""

from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import sys
import os
import pandas as pd
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.data.data_handler import DataLoader

# Load environment variables
load_dotenv()


def initialize_system():
    """Initialize base LLM and load datasets"""

    print("\n" + "=" * 80)
    print(" " * 25 + "BASELINE LLM INITIALIZATION")
    print("=" * 80)

    print("\n Initializing AWS Bedrock LLM (base model, no agents)...")
    try:
        llm = ChatBedrock(
            model_id="meta.llama3-1-70b-instruct-v1:0",
            model_kwargs={
                "temperature": 0.1,
            }
        )
        print(" Bedrock LLM initialized successfully")
        print("   Model: Llama 3.1 70B Instruct")
        print("   Context Window: 128K tokens")
        print("   Temperature: 0.1")
        print("   Mode: Direct LLM invocation (no agent system)")
    except Exception as e:
        print(f" Error initializing Bedrock LLM: {e}")
        print("\nTroubleshooting:")
        print("1. If using temporary credentials (ASIA*), add AWS_SESSION_TOKEN to .env")
        print("2. If using permanent credentials, ensure AWS_ACCESS_KEY_ID starts with AKIA")
        print("3. Or configure AWS CLI: aws configure")
        print("4. Ensure your AWS account has Bedrock access in the selected region")
        sys.exit(1)

    print("\n Loading datasets...")
    loader = DataLoader()
    datasets = {}

    try:
        housing_path = "data/housing.csv"
        datasets['housing'] = loader.load_data(housing_path)
        datasets['housing'] = datasets['housing']
        print(f" Loaded housing dataset: {len(datasets['housing'])} rows, {len(datasets['housing'].columns)} columns")
    except Exception as e:
        print(f" Warning: Could not load housing dataset: {e}")

    try:
        coffee_path = "data/coffee_shop_sales.xlsx"
        datasets['coffee'] = loader.load_data(coffee_path)
        datasets['coffee'] = datasets['coffee'].iloc[0:1000]
        print(f" Loaded coffee dataset: {len(datasets['coffee'])} rows, {len(datasets['coffee'].columns)} columns")
    except Exception as e:
        print(f" Warning: Could not load coffee dataset: {e}")

    return llm, datasets


def load_baseline_questions(filepath="data/baseline_describe_data.csv"):
    """Load baseline questions from CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"\n Loaded {len(df)} baseline questions from {filepath}")
        return df
    except Exception as e:
        print(f" Error loading baseline questions: {e}")
        sys.exit(1)


def format_dataset_for_prompt(df, dataset_name):
    """Format dataset as string representation for LLM"""

    # Return dataset as string for LLM
    info = f"\n{dataset_name.upper()} DATASET:\n"
    info += df.to_string(index=False)
    info += "\n\n"

    return info


def run_baseline_tests(llm, baseline_df, datasets):
    """Run each baseline question through the base LLM with dataset context"""

    results = []
    total = len(baseline_df)

    print("\n" + "=" * 80)
    print(" " * 25 + "RUNNING BASELINE TESTS")
    print("=" * 80)
    print(f"\nProcessing {total} questions with base LLM (no agent system)...\n")

    # Format dataset information once
    dataset_context = ""
    if 'housing' in datasets:
        dataset_context += format_dataset_for_prompt(datasets['housing'], 'Housing')
    if 'coffee' in datasets:
        dataset_context += format_dataset_for_prompt(datasets['coffee'], 'Coffee Shop Sales')

    for idx, row in baseline_df.iterrows():
        user_request = row['user_request']
        ground_truth = row['ground_truth']

        print(f"[{idx + 1}/{total}] Testing: {user_request[:60]}...")

        try:
            full_prompt = f"""Here are the datasets:

{dataset_context}

Question: {user_request}

Please answer the question based on the datasets above. Keep the answers short and consise."""

            start_time = time.time()

            llm_response = llm.invoke(full_prompt)

            execution_time = time.time() - start_time

            if hasattr(llm_response, 'content'):
                response_text = llm_response.content
            else:
                response_text = str(llm_response)

            results.append({
                'user_request': user_request,
                'ground_truth': ground_truth,
                'llm_response': response_text,
                'execution_time_seconds': round(execution_time, 2)
            })

            print(f" Completed in {execution_time:.2f}s")

        except Exception as e:
            print(f" Error: {e}")
            results.append({
                'user_request': user_request,
                'ground_truth': ground_truth,
                'llm_response': f"ERROR: {str(e)}",
                'execution_time_seconds': 0
            })

    return results


def save_results(results):
    """Save results to CSV with timestamp in filename"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/baseline_results_{timestamp}.csv"

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\n Results saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        sys.exit(1)


def print_summary(results):
    """Print summary statistics"""
    total = len(results)
    errors = sum(1 for r in results if r['llm_response'].startswith('ERROR:'))
    successful = total - errors

    execution_times = [r['execution_time_seconds'] for r in results if r['execution_time_seconds'] > 0]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        total_time = sum(execution_times)
    else:
        avg_time = min_time = max_time = total_time = 0

    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal Questions: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    print(f"\nExecution Time Statistics:")
    print(f" Average: {avg_time:.2f}s per request")
    print(f" Minimum: {min_time:.2f}s")
    print(f" Maximum: {max_time:.2f}s")
    print(f" Total: {total_time:.2f}s")
    print("\n" + "=" * 80)


def main():
    """Main execution function"""

    start_time = datetime.now()

    llm, datasets = initialize_system()

    baseline_df = load_baseline_questions()
    results = run_baseline_tests(llm, baseline_df, datasets)

    output_path = save_results(results)

    print_summary(results)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n Total execution time: {duration:.2f} seconds")
    print(f" Results saved to: {output_path}")
    print(f"\n This baseline test used the raw LLM with dataset context but no agent system.")
    print(f" Compare these results with agent-based results to measure improvement.\n")


if __name__ == "__main__":
    main()