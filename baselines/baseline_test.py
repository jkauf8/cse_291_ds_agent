"""
Baseline testing script for evaluating base LLM performance.
Tests Gemini LLM without any agent system to establish baseline performance.
Uses LLM as a judge to evaluate ground truth accuracy.
"""

from dotenv import load_dotenv
import sys
import os
import pandas as pd
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini_llm import GeminiLLM
from prompts.llm_as_a_judge import llm_judge_prompt

# Load environment variables
load_dotenv()


def initialize_system():
    """Initialize Gemini LLM and load housing dataset"""

    print("\n" + "=" * 80)
    print(" " * 25 + "BASELINE LLM INITIALIZATION")
    print("=" * 80)

    print("\n Initializing Gemini LLM (base model, no agents)...")
    try:
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )
        print(" Gemini LLM initialized successfully")
        print("   Model: Gemini 2.5 Flash")
        print("   Temperature: 0.1")
        print("   Mode: Direct LLM invocation (no agent system)")
    except Exception as e:
        print(f" Error initializing Gemini LLM: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure GEMINI_API_KEY is set in your .env file")
        print("2. Verify your Gemini API key is valid")
        sys.exit(1)

    print("\n Loading housing dataset...")
    try:
        housing_df = pd.read_csv("data/housing.csv")
        print(f" Loaded housing dataset: {len(housing_df)} rows, {len(housing_df.columns)} columns")
    except Exception as e:
        print(f" Error: Could not load housing dataset: {e}")
        sys.exit(1)

    return llm, housing_df


def load_validation_questions(filepath="validation/final_validation.csv"):
    """Load validation questions from CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"\n Loaded {len(df)} validation questions from {filepath}")
        return df
    except Exception as e:
        print(f" Error loading validation questions: {e}")
        sys.exit(1)


def format_dataset_for_prompt(df):
    """Format dataset as string representation for LLM"""
    # Return dataset as string for LLM
    info = df.to_string(index=False)
    return info


def judge_responses_with_llm(llm, results):
    """
    Use LLM as a judge to evaluate each response against ground truth.
    Updates results in-place with ground_truth_match field.
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
                "model_response": result['llm_response']
            })

            # Extract the judgment (1 or 0)
            judgment_text = judgment if isinstance(judgment, str) else str(judgment)

            # Parse the score
            try:
                score = int(judgment_text.strip())
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


def run_baseline_tests(llm, baseline_df, housing_df):
    """Run each validation question through Gemini LLM with housing dataset context"""

    results = []
    total = len(baseline_df)

    print("\n" + "=" * 80)
    print(" " * 25 + "RUNNING BASELINE TESTS")
    print("=" * 80)
    print(f"\nProcessing {total} questions with Gemini LLM (no agent system)...\n")

    # Format dataset information once
    dataset_context = format_dataset_for_prompt(housing_df)

    for idx, row in baseline_df.iterrows():
        user_request = row['user_request']
        ground_truth = row.get('ground_truth', '')

        print(f"[{idx + 1}/{total}] Testing: {user_request[:60]}...")

        try:
            full_prompt = f"""You are a data analysis assistant. You have access to a housing dataset.

{dataset_context}

User Question: {user_request}

Please provide a clear and concise answer based on the dataset information provided above."""

            start_time = time.time()

            # Use _call method for direct LLM invocation
            response_text = llm._call(full_prompt)

            execution_time = time.time() - start_time

            results.append({
                'user_request': user_request,
                'ground_truth': ground_truth,
                'llm_response': response_text,
                'ground_truth_match': None,  # Will be filled by judge
                'judge_score': None,  # Will be filled by judge
                'execution_time_seconds': round(execution_time, 2)
            })

            print(f" Completed in {execution_time:.2f}s")

        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'user_request': user_request,
                'ground_truth': ground_truth,
                'llm_response': f"ERROR: {str(e)}",
                'ground_truth_match': False,
                'judge_score': 0,
                'execution_time_seconds': 0
            })

    return results


def save_results(results):
    """Save results to CSV with timestamp in filename"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"baselines/baseline_results_{timestamp}.csv"

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\n Results saved to {output_path}")
        return output_path
    except Exception as e:
        print(f" Error saving results: {e}")
        sys.exit(1)


def print_summary(results):
    """Print summary statistics"""
    total = len(results)
    errors = sum(1 for r in results if str(r['llm_response']).startswith('ERROR:'))
    successful = total - errors

    # Ground Truth Accuracy
    results_with_gt = [r for r in results if r.get('ground_truth') and
                      not pd.isna(r.get('ground_truth')) and
                      r.get('ground_truth') != '']

    if results_with_gt:
        gt_correct_count = sum(1 for r in results_with_gt if r.get('ground_truth_match', False))
        gta = (gt_correct_count / len(results_with_gt)) * 100
    else:
        gt_correct_count = 0
        gta = 0

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

    print("\n" + "=" * 80)
    print(" " * 30 + "BASELINE SUMMARY")
    print("=" * 80)
    print(f"\nTotal Questions: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")

    print(f"\n--- GROUND TRUTH ACCURACY (GTA) ---")
    print(f"Ground Truth Accuracy (via LLM Judge): {gta:.1f}%")
    print(f"Correct Ground Truth Matches: {gt_correct_count}/{len(results_with_gt)}")
    print(f"Average Judge Score: {avg_judge_score:.2f}")

    print(f"\n--- Execution Time Statistics ---")
    print(f" Average: {avg_time:.2f}s per request")
    print(f" Minimum: {min_time:.2f}s")
    print(f" Maximum: {max_time:.2f}s")
    print(f" Total: {total_time:.2f}s")
    print("\n" + "=" * 80)


def main():
    """Main execution function"""

    start_time = datetime.now()

    llm, housing_df = initialize_system()
    baseline_df = load_validation_questions()

    results = run_baseline_tests(llm, baseline_df, housing_df)

    # Use LLM as judge to evaluate responses
    judge_responses_with_llm(llm, results)

    output_path = save_results(results)
    print_summary(results)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n Total execution time: {duration:.2f} seconds")
    print(f" Results saved to: {output_path}")
    print(f"\n This baseline test used raw Gemini LLM with dataset context but no agent system.")
    print(f" Compare these results with agent-based results to measure improvement.\n")


if __name__ == "__main__":
    main()