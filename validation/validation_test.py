from dotenv import load_dotenv
import sys
import os
import pandas as pd
from datetime import datetime
import time
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_graph import AgentGraph
from prompts.llm_as_a_judge import llm_judge_prompt
from gemini_llm import GeminiLLM
from langchain_aws import ChatBedrock

# Load environment variables
load_dotenv()


def initialize_system(use_bedrock=False):
    """Initialize AgentGraph system and load datasets (same as main.py)"""

    print("\n" + "=" * 80)
    print(" " * 25 + "VALIDATION TEST INITIALIZATION")
    print("=" * 80)

    if use_bedrock:
        print("\n Initializing AWS Bedrock LLM for Agent...")
        try:
            llm = ChatBedrock(
                model_id="meta.llama3-1-70b-instruct-v1:0",
                model_kwargs={
                    "temperature": 0.1,
                }
            )
            print(" Bedrock LLM initialized successfully")
            print(" Model: Llama 3.1 70B Instruct")
        except Exception as e:
            print(f" Error initializing Bedrock LLM: {e}")
            sys.exit(1)
    else:
        print("\n Initializing Gemini LLM for Agent...")
        try:
            llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
            print(" Gemini LLM initialized successfully")
            print(" Model: Gemini 2.5 Flash")
        except Exception as e:
            print(f" Error initializing Gemini LLM: {e}")
            sys.exit(1)

    print("\n Loading datasets...")
    try:
        housing_df = pd.read_csv('data/housing.csv')
        print(f" Loaded housing dataset: {len(housing_df)} rows, {len(housing_df.columns)} columns")
    except Exception as e:
        print(f" Error: Could not load housing dataset: {e}")
        sys.exit(1)

    datasets = {"housing": housing_df}

    print("\n Initializing AgentGraph...")
    try:
        agent_graph = AgentGraph(llm=llm, datasets=datasets)
        print(" AgentGraph initialized successfully")
    except Exception as e:
        print(f" Error initializing AgentGraph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if use_bedrock:
        print("\n Initializing Bedrock LLM for Judge...")
        try:
            judge_llm = ChatBedrock(
                model_id="meta.llama3-1-70b-instruct-v1:0",
                model_kwargs={
                    "temperature": 0.1,
                }
            )
            print(" Bedrock Judge LLM initialized successfully")
        except Exception as e:
            print(f" Error initializing Bedrock Judge LLM: {e}")
            sys.exit(1)
    else:
        print("\n Initializing Gemini LLM for Judge...")
        try:
            judge_llm = GeminiLLM(model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
            print(" Gemini Judge LLM initialized successfully")
        except Exception as e:
            print(f" Error initializing Gemini Judge LLM: {e}")
            sys.exit(1)

    return agent_graph, judge_llm


def normalize_tool_name(tool_name):
    """
    Normalize tool names for comparison.
    If it's a list with both describe and regression, return 'both' to match CSV.
    """
    # Check if it's a list FIRST, before doing pd.isna() or string comparison
    if isinstance(tool_name, list):
        has_describe = any('describe' in str(t).lower() for t in tool_name)
        has_regression = any('regression' in str(t).lower() or 'regress' in str(t).lower() for t in tool_name)
        if has_describe and has_regression:
            return 'both'
        # If only one type, return the single normalized tool
        elif has_describe:
            return 'describe_data()'
        elif has_regression:
            return 'run_regression()'

    # Now check for None/empty after handling lists
    if pd.isna(tool_name) or tool_name == "":
        return None

    tool_name = str(tool_name).strip().lower()

    # If already "both", keep it as both
    if tool_name == 'both':
        return 'both'

    # Standardize single tool names
    if 'describe' in tool_name:
        return 'describe_data()'
    elif 'regression' in tool_name or 'regress' in tool_name:
        return 'run_regression()'
    elif 'direct' in tool_name or 'response' in tool_name:
        return 'direct_response()'

    return tool_name


def judge_responses_with_llm(llm, results):
    """
    Use LLM as a judge to evaluate each response against ground truth.
    Updates results in-place with ground_truth_match field.
    Returns number of results judged.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "LLM JUDGE EVALUATION")
    print("=" * 80)

    # Filter results that have ground truth
    results_to_judge = [r for r in results if r.get('ground_truth') and
                       not pd.isna(r.get('ground_truth')) and
                       r.get('ground_truth') != '']

    total_to_judge = len(results_to_judge)
    print(f"\nEvaluating {total_to_judge} responses with ground truth using LLM judge...\n")

    judged_count = 0

    for idx, result in enumerate(results):
        # Skip if no ground truth
        if not result.get('ground_truth') or pd.isna(result.get('ground_truth')) or result.get('ground_truth') == '':
            result['ground_truth_match'] = None
            result['judge_score'] = None
            continue

        print(f"[{idx + 1}/{total_to_judge}] Judging response for: {result['user_request'][:50]}...")

        try:
            # Create the judge prompt
            judge_chain = llm_judge_prompt | llm

            # Get LLM judgment
            judgment = judge_chain.invoke({
                "user_request": result['user_request'],
                "ground_truth": result['ground_truth'],
                "model_response": result['agent_response']
            })

            # Extract the judgment (1 or 0)
            judgment_text = judgment.content.strip() if hasattr(judgment, 'content') else str(judgment).strip()

            # Parse the score
            try:
                score = int(judgment_text)
                if score not in [0, 1]:
                    print(f"  Warning: Judge returned non-binary score: {judgment_text}, defaulting to 0")
                    score = 0
            except ValueError:
                print(f"  Warning: Could not parse judge score: {judgment_text}, defaulting to 0")
                score = 0

            result['ground_truth_match'] = (score == 1)
            result['judge_score'] = score

            status = "CORRECT" if score == 1 else "INCORRECT"
            print(f"  Judge: {status}")
            judged_count += 1

        except Exception as e:
            print(f"  Error during judging: {e}")
            result['ground_truth_match'] = False
            result['judge_score'] = 0

    print(f"\n Completed judging {judged_count} responses")
    print("=" * 80)

    return judged_count


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

        print(f"[{idx + 1}/{total}] Testing: {user_request[:60]}...")

        try:
            start_time = time.time()

            # Run the agent graph
            final_state = agent_graph.run(user_request)

            execution_time = time.time() - start_time

            # Extract results from state
            # Use selected_tools which persists from planner, not route which gets overwritten by reviewer
            predicted_tool = final_state.get('selected_tools', 'unknown')
            response_text = final_state.get('response', '')

            # Track all tools used throughout execution
            tools_used = predicted_tool if isinstance(predicted_tool, list) else [predicted_tool]
            tools_used_str = ', '.join(str(tool) for tool in tools_used)

            # Normalize tool names for comparison
            # normalize_tool_name handles lists and returns 'both' if both tools present
            normalized_predicted = normalize_tool_name(predicted_tool)
            normalized_ground_truth = normalize_tool_name(ground_truth_tool)

            # Check if tool selection was correct (simple string comparison now)
            tool_correct = (normalized_predicted == normalized_ground_truth)

            # Ground truth matching will be done by LLM judge later
            results.append({
                'user_request': user_request,
                'ground_truth_tool': ground_truth_tool,
                'tools_used': tools_used_str,
                'tool_correct': tool_correct,
                'ground_truth': ground_truth,
                'agent_response': response_text,
                'ground_truth_match': None,  # Will be filled by LLM judge
                'judge_score': None,  # Will be filled by LLM judge
                'execution_time_seconds': round(execution_time, 2)
            })

            status = ""
            if not tool_correct:
                status += " [TOOL MISMATCH]"

            print(f" Completed in {execution_time:.2f}s - Tool: {predicted_tool}{status}")

        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'user_request': user_request,
                'ground_truth_tool': ground_truth_tool,
                'tools_used': 'ERROR',
                'tool_correct': False,
                'ground_truth': ground_truth,
                'agent_response': f"ERROR: {str(e)}",
                'ground_truth_match': False,
                'judge_score': 0,
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
    """Calculate Tool Selection Accuracy, Ground Truth Accuracy, and other metrics"""

    total = len(results)
    errors = sum(1 for r in results if r['tools_used'] == 'ERROR')
    successful = total - errors

    # Tool Selection Accuracy (TSA)
    tool_correct_count = sum(1 for r in results if r['tool_correct'])
    tsa = (tool_correct_count / total) * 100 if total > 0 else 0

    # Ground Truth Accuracy (GTA)
    # Only calculate for cases where ground truth exists
    results_with_gt = [r for r in results if r.get('ground_truth') and
                      not pd.isna(r.get('ground_truth')) and
                      r.get('ground_truth') != '']

    if results_with_gt:
        gt_correct_count = sum(1 for r in results_with_gt if r.get('ground_truth_match', False))
        gta = (gt_correct_count / len(results_with_gt)) * 100
    else:
        gt_correct_count = 0
        gta = 0

    # Breakdown by tool type
    tool_breakdown = {}
    for r in results:
        gt_tool = r['ground_truth_tool']
        if gt_tool not in tool_breakdown:
            tool_breakdown[gt_tool] = {'total': 0, 'correct': 0, 'gt_correct': 0, 'both_correct': 0}
        tool_breakdown[gt_tool]['total'] += 1
        if r['tool_correct']:
            tool_breakdown[gt_tool]['correct'] += 1
        if r.get('ground_truth_match', False):
            tool_breakdown[gt_tool]['gt_correct'] += 1
        if r.get('tool_correct', False) and r.get('ground_truth_match', False):
            tool_breakdown[gt_tool]['both_correct'] += 1

    # Calculate average judge score (for informational purposes)
    judge_scores = [r.get('judge_score', 0) for r in results_with_gt if r.get('judge_score') is not None]
    avg_judge_score = (sum(judge_scores) / len(judge_scores)) if judge_scores else 0

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
        'gta': gta,
        'gt_correct_count': gt_correct_count,
        'results_with_gt_count': len(results_with_gt),
        'avg_judge_score': avg_judge_score,
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

    print(f"\n--- GROUND TRUTH ACCURACY (GTA) ---")
    print(f"Ground Truth Accuracy (via LLM Judge): {metrics['gta']:.1f}%")
    print(f"Correct Ground Truth Matches: {metrics['gt_correct_count']}/{metrics['results_with_gt_count']}")
    print(f"Average Judge Score: {metrics['avg_judge_score']:.2f}")

    print(f"\n--- Tool Breakdown ---")
    for tool, stats in metrics['tool_breakdown'].items():
        tool_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        gt_acc = (stats['gt_correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        both_acc = (stats['both_correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"\n{tool}:")
        print(f"  Tool Selection: {stats['correct']}/{stats['total']} ({tool_acc:.1f}%)")
        print(f"  Ground Truth: {stats['gt_correct']}/{stats['total']} ({gt_acc:.1f}%)")
        print(f"  Both Correct: {stats['both_correct']}/{stats['total']} ({both_acc:.1f}%)")

    print(f"\n--- Execution Time Statistics ---")
    print(f" Average: {metrics['avg_time']:.2f}s per request")
    print(f" Minimum: {metrics['min_time']:.2f}s")
    print(f" Maximum: {metrics['max_time']:.2f}s")
    print(f" Total: {metrics['total_time']:.2f}s")
    print("\n" + "=" * 80)


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validation testing for agent system")
    parser.add_argument(
        '--bedrock',
        action='store_true',
        help='Use AWS Bedrock Llama instead of Gemini'
    )
    args = parser.parse_args()

    start_time = datetime.now()

    agent_graph, llm = initialize_system(use_bedrock=args.bedrock)
    validation_df = pd.read_csv("data/final_validation.csv")
    # validation_df = validation_df.iloc[13:16]
    # validation_df.reset_index(inplace=True)

    results = run_validation_tests(agent_graph, validation_df)

    # Use LLM as a judge to evaluate responses against ground truth
    judge_responses_with_llm(llm, results)

    output_path = save_results(results)
    metrics = calculate_accuracy_metrics(results)
    print_summary(metrics)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n Total execution time: {duration:.2f} seconds")
    print(f" Results saved to: {output_path}")

    llm_type = "Bedrock Llama 3.1 70B" if args.bedrock else "Gemini 2.5 Flash"
    print(f" LLM: {llm_type} (both agent and judge)")
    print(f" Tool Selection Accuracy (TSA): {metrics['tsa']:.1f}%")
    print(f" Ground Truth Accuracy (GTA): {metrics['gta']:.1f}%")


if __name__ == "__main__":
    main()
