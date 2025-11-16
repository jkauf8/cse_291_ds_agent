"""
Validation testing script for evaluating agent system performance.
Tests the full AgentGraph system on validation questions and tracks:
- Tool selection accuracy (TSA)
- Response quality against ground truth
- Execution times
"""

from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import sys
import os
import pandas as pd
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_graph import AgentGraph
from data_handler import DataLoader

# Load environment variables
load_dotenv()


def initialize_system():
    """Initialize AgentGraph system and load datasets"""

    print("\n" + "=" * 80)
    print(" " * 25 + "VALIDATION TEST INITIALIZATION")
    print("=" * 80)

    print("\n Initializing AWS Bedrock LLM...")
    try:
        llm = ChatBedrock(
            model_id="meta.llama3-1-70b-instruct-v1:0",
            model_kwargs={
                "temperature": 0.1,
                # "max_tokens": 2048
            }
        )
        print(" Bedrock LLM initialized successfully")
        print(" Model: Llama 3.1 70B Instruct")
        print(" Temperature: 0.1")
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

    print("\n Initializing AgentGraph...")
    try:
        agent_graph = AgentGraph(llm=llm, datasets=datasets)
        print(" AgentGraph initialized successfully")
        return agent_graph
    except Exception as e:
        print(f" Error initializing AgentGraph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_validation_questions(filepath="data/validation.csv"):
    """Load validation questions from CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"\n Loaded {len(df)} validation questions from {filepath}")
        return df
    except Exception as e:
        print(f" Error loading validation questions: {e}")
        sys.exit(1)


def normalize_tool_name(tool_name):
    """Normalize tool names for comparison"""
    if pd.isna(tool_name) or tool_name == "":
        return None

    tool_name = str(tool_name).strip().lower()

    # Standardize tool names
    if 'describe' in tool_name:
        return 'describe_data()'
    elif 'regression' in tool_name or 'regress' in tool_name:
        return 'run_regression()'
    elif 'direct' in tool_name or 'response' in tool_name:
        return 'direct_response()'

    return tool_name


def run_validation_tests(agent_graph, validation_df):
    """Run each validation question through the AgentGraph system"""

    results = []
    total = len(validation_df)

    print("\n" + "=" * 80)
    print(" " * 25 + "RUNNING VALIDATION TESTS")
    print("=" * 80)
    print(f"\nProcessing {total} questions with full AgentGraph system...\n")

    for idx, row in validation_df.iterrows():
        user_request = row['user_request']
        ground_truth_tool = row.get('ground_truth_tool', '')
        ground_truth = row.get('ground_truth', '')
        expected_criteria = row.get('expected_report_criteria', '')

        print(f"[{idx + 1}/{total}] Testing: {user_request[:60]}...")

        try:
            start_time = time.time()

            # Run the agent graph
            final_state = agent_graph.run(user_request)

            execution_time = time.time() - start_time

            # Extract results from state
            predicted_tool = final_state.get('route', {}).get('router_decision', 'unknown')
            response_text = final_state.get('response', '')

            # Normalize tool names for comparison
            normalized_predicted = normalize_tool_name(predicted_tool)
            normalized_ground_truth = normalize_tool_name(ground_truth_tool)

            # Check if tool selection was correct
            tool_correct = (normalized_predicted == normalized_ground_truth)

            results.append({
                'user_request': user_request,
                'ground_truth_tool': ground_truth_tool,
                'predicted_tool': predicted_tool,
                'tool_correct': tool_correct,
                'ground_truth': ground_truth,
                'agent_response': response_text,
                'expected_criteria': expected_criteria,
                'execution_time_seconds': round(execution_time, 2)
            })

            status = "" if tool_correct else " [TOOL MISMATCH]"
            print(f" Completed in {execution_time:.2f}s - Tool: {predicted_tool}{status}")

        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'user_request': user_request,
                'ground_truth_tool': ground_truth_tool,
                'predicted_tool': 'ERROR',
                'tool_correct': False,
                'ground_truth': ground_truth,
                'agent_response': f"ERROR: {str(e)}",
                'expected_criteria': expected_criteria,
                'execution_time_seconds': 0
            })

    return results


def save_results(results):
    """Save results to CSV with timestamp in filename"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"validation/validation_results_{timestamp}.csv"

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\n Results saved to {output_path}")
        return output_path
    except Exception as e:
        print(f" Error saving results: {e}")
        sys.exit(1)


def calculate_accuracy_metrics(results):
    """Calculate Tool Selection Accuracy and other metrics"""

    total = len(results)
    errors = sum(1 for r in results if r['predicted_tool'] == 'ERROR')
    successful = total - errors

    # Tool Selection Accuracy (TSA)
    tool_correct_count = sum(1 for r in results if r['tool_correct'])
    tsa = (tool_correct_count / total) * 100 if total > 0 else 0

    # Breakdown by tool type
    tool_breakdown = {}
    for r in results:
        gt_tool = r['ground_truth_tool']
        if gt_tool not in tool_breakdown:
            tool_breakdown[gt_tool] = {'total': 0, 'correct': 0}
        tool_breakdown[gt_tool]['total'] += 1
        if r['tool_correct']:
            tool_breakdown[gt_tool]['correct'] += 1

    # Execution time statistics
    execution_times = [r['execution_time_seconds'] for r in results if r['execution_time_seconds'] > 0]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        total_time = sum(execution_times)
    else:
        avg_time = min_time = max_time = total_time = 0

    return {
        'total': total,
        'successful': successful,
        'errors': errors,
        'tsa': tsa,
        'tool_correct_count': tool_correct_count,
        'tool_breakdown': tool_breakdown,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'total_time': total_time
    }


def print_summary(metrics):
    """Print summary statistics and accuracy metrics"""

    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)

    print(f"\nTotal Questions: {metrics['total']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Success Rate: {(metrics['successful']/metrics['total'])*100:.1f}%")

    print(f"\n--- TOOL SELECTION ACCURACY (TSA) ---")
    print(f"Tool Selection Accuracy: {metrics['tsa']:.1f}%")
    print(f"Correct Tool Selections: {metrics['tool_correct_count']}/{metrics['total']}")

    print(f"\n--- Tool Breakdown ---")
    for tool, stats in metrics['tool_breakdown'].items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{tool}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

    print(f"\n--- Execution Time Statistics ---")
    print(f" Average: {metrics['avg_time']:.2f}s per request")
    print(f" Minimum: {metrics['min_time']:.2f}s")
    print(f" Maximum: {metrics['max_time']:.2f}s")
    print(f" Total: {metrics['total_time']:.2f}s")
    print("\n" + "=" * 80)


def main():
    """Main execution function"""

    start_time = datetime.now()

    agent_graph = initialize_system()
    validation_df = load_validation_questions()
    results = run_validation_tests(agent_graph, validation_df)

    output_path = save_results(results)
    metrics = calculate_accuracy_metrics(results)
    print_summary(metrics)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n Total execution time: {duration:.2f} seconds")
    print(f" Results saved to: {output_path}")
    print(f"\n This validation test used the full AgentGraph system.")
    print(f" Tool Selection Accuracy (TSA): {metrics['tsa']:.1f}%\n")


if __name__ == "__main__":
    main()
