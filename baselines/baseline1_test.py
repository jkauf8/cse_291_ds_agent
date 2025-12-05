from dotenv import load_dotenv
import sys
import os
import pandas as pd
from datetime import datetime
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini_llm import GeminiLLM
from langchain_aws import ChatBedrock
from prompts.llm_as_a_judge import llm_judge_prompt

# Load environment variables
load_dotenv()


def initialize_system(use_bedrock=False, max_tokens=1024):
    """Initialize LLM (Gemini or Bedrock) and load housing dataset"""

    print("BASELINE LLM INITIALIZATION")

    if use_bedrock:
        print("\nInitializing AWS Bedrock LLM (base model, no agents)...")
        try:
            llm = ChatBedrock(
                model_id="meta.llama3-1-70b-instruct-v1:0",
                model_kwargs={
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                }
            )
            print("Bedrock LLM initialized successfully")
            print("Model: Llama 3.1 70B Instruct")
            print("Temperature: 0.1")
            print(f"Max Tokens: {max_tokens}")
            print("Mode: Direct LLM invocation (no agent system)")
        except Exception as e:
            print(f"Error initializing Bedrock LLM: {e}")
            print("Troubleshooting:")
            print("1. Ensure AWS credentials are set in your .env file")
            print("2. Verify your AWS credentials have Bedrock access")
            sys.exit(1)
    else:
        print("Initializing Gemini LLM (base model, no agents)...")
        try:
            llm = GeminiLLM(
                model_name="gemini-2.5-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
                max_output_tokens=max_tokens
            )
            print("Gemini LLM initialized successfully")
            print("Model: Gemini 2.5 Flash")
            print("Temperature: 0.1")
            print(f"Max Output Tokens: {max_tokens}")
            print("Mode: Direct LLM invocation (no agent system)")
        except Exception as e:
            print(f"Error initializing Gemini LLM: {e}")
            print("Troubleshooting:")
            print("1. Ensure GEMINI_API_KEY is set in your .env file")
            print("2. Verify your Gemini API key is valid")
            sys.exit(1)

    print("\nLoading housing dataset...")
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        housing_path = os.path.join(script_dir, "data", "housing.csv")
        housing_df = pd.read_csv(housing_path)
        print(f"Loaded housing dataset: {len(housing_df)} rows, {len(housing_df.columns)} columns")
    except Exception as e:
        print(f"Error: Could not load housing dataset: {e}")
        sys.exit(1)

    return llm, housing_df


def load_validation_questions(filepath="data/final_validation.csv"):
    """Load validation questions from CSV"""
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(script_dir, filepath)
        df = pd.read_csv(full_path)
        # df = df.iloc[0:2]
        print(f"Loaded {len(df)} validation questions from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading validation questions: {e}")
        sys.exit(1)


def format_dataset_for_prompt(df):
    """Format dataset as string representation for LLM"""
    info = df.to_string(index=False)
    return info


def judge_responses_with_llm(llm, results):
    """
    Use LLM as a judge to evaluate each response against ground truth.
    Updates results in-place with ground_truth_match field.
    """
    print("LLM JUDGE EVALUATION")

    results_to_judge = [r for r in results if r.get('ground_truth') and
                       not pd.isna(r.get('ground_truth')) and
                       r.get('ground_truth') != '']

    total_to_judge = len(results_to_judge)
    print(f"Evaluating {total_to_judge} responses with ground truth using LLM judge...\n")

    judged_count = 0

    for idx, result in enumerate(results):
        # Skip if no ground truth
        if not result.get('ground_truth') or pd.isna(result.get('ground_truth')) or result.get('ground_truth') == '':
            result['ground_truth_match'] = None
            result['judge_score'] = None
            continue

        print(f"[{idx + 1}/{total_to_judge}] Judging response for: {result['user_request'][:50]}...")

        try:
            judge_chain = llm_judge_prompt | llm

            judgment = judge_chain.invoke({
                "user_request": result['user_request'],
                "ground_truth": result['ground_truth'],
                "model_response": result['llm_response']
            })

            if hasattr(judgment, 'content'):
                judgment_text = judgment.content
            elif isinstance(judgment, str):
                judgment_text = judgment
            else:
                judgment_text = str(judgment)

            try:
                score = int(judgment_text.strip())
                if score not in [0, 1]:
                    print(f"Warning: Judge returned non-binary score: {judgment_text}, defaulting to 0")
                    score = 0
            except ValueError:
                print(f"Warning: Could not parse judge score: {judgment_text}, defaulting to 0")
                score = 0

            result['ground_truth_match'] = (score == 1)
            result['judge_score'] = score

            status = "CORRECT" if score == 1 else "INCORRECT"
            print(f"Judge: {status}")
            judged_count += 1

        except Exception as e:
            print(f"Error during judging: {e}")
            result['ground_truth_match'] = False
            result['judge_score'] = 0

    print("Completed judging {judged_count} responses")

    return judged_count


def run_baseline_tests(llm, baseline_df, housing_df, use_bedrock=False):
    """Run each validation question through LLM with housing dataset context"""

    results = []
    total = len(baseline_df)
    timeout_seconds = 120

    print("RUNNING BASELINE TESTS")

    llm_name = "Bedrock Llama" if use_bedrock else "Gemini"
    print(f"Processing {total} questions with {llm_name} LLM (no agent system)...")
    print(f"Timeout: {timeout_seconds}s per question")

    dataset_context = format_dataset_for_prompt(housing_df)

    def call_llm(prompt, use_bedrock_flag):
        """Helper function to call LLM (for timeout handling)"""
        if use_bedrock_flag:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            return llm._call(prompt)

    executor = ThreadPoolExecutor(max_workers=1)

    try:
        for idx, row in baseline_df.iterrows():
            user_request = row['user_request']
            ground_truth = row.get('ground_truth', '')

            print(f"[{idx + 1}/{total}] Testing: {user_request[:60]}...")

            try:
                full_prompt = f"""You are a data analysis assistant. You have access to a housing dataset.

{dataset_context}

User Question: {user_request}

IMPORTANT: Provide a BRIEF, CONCISE answer (maximum 2-3 sentences). Do not write code. Just describe what you would do or what the answer is based on the data."""

                start_time = time.time()

                future = executor.submit(call_llm, full_prompt, use_bedrock)
                try:
                    response_text = future.result(timeout=timeout_seconds)
                    execution_time = time.time() - start_time

                    results.append({
                        'user_request': user_request,
                        'ground_truth': ground_truth,
                        'llm_response': response_text,
                        'ground_truth_match': None,
                        'judge_score': None,
                        'execution_time_seconds': round(execution_time, 2)
                    })

                    print(f" Completed in {execution_time:.2f}s")

                except FuturesTimeoutError:
                    execution_time = time.time() - start_time
                    print(f" TIMEOUT after {timeout_seconds}s - canceling and moving to next request")

                    future.cancel()

                    results.append({
                        'user_request': user_request,
                        'ground_truth': ground_truth,
                        'llm_response': f"TIMEOUT: LLM did not respond within {timeout_seconds} seconds",
                        'ground_truth_match': False,
                        'judge_score': 0,
                        'execution_time_seconds': round(execution_time, 2)
                    })

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
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


def save_results(results):
    """Save results to CSV with timestamp in filename"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, f"baseline_results_{timestamp}.csv")

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def print_summary(results):
    """Print summary statistics"""
    total = len(results)
    timeouts = sum(1 for r in results if str(r['llm_response']).startswith('TIMEOUT:'))
    errors = sum(1 for r in results if str(r['llm_response']).startswith('ERROR:'))
    successful = total - errors - timeouts

    # GTA
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

    execution_times = [r['execution_time_seconds'] for r in results if r['execution_time_seconds'] > 0]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        total_time = sum(execution_times)
    else:
        avg_time = min_time = max_time = total_time = 0
    
    print("\n")
    print("BASELINE SUMMARY")
    print(f"Total Questions: {total}")
    print(f"Successful: {successful}")
    print(f"Timeouts: {timeouts}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    print("\n")
    print(f"GROUND TRUTH ACCURACY (GTA)")
    print(f"Ground Truth Accuracy (via LLM Judge): {gta:.1f}%")
    print(f"Correct Ground Truth Matches: {gt_correct_count}/{len(results_with_gt)}")
    print(f"Average Judge Score: {avg_judge_score:.2f}")
    print("\n")
    print(f"Execution Time Statistics")
    print(f"Average: {avg_time:.2f}s per request")
    print(f"Minimum: {min_time:.2f}s")
    print(f"Maximum: {max_time:.2f}s")
    print(f"Total: {total_time:.2f}s")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Baseline testing for LLM performance")
    parser.add_argument(
        '--bedrock',
        action='store_true',
        help='Use AWS Bedrock Llama instead of Gemini'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum number of tokens to generate per response (default: 512)'
    )
    args = parser.parse_args()

    start_time = datetime.now()

    llm, housing_df = initialize_system(use_bedrock=args.bedrock, max_tokens=args.max_tokens)
    baseline_df = load_validation_questions()

    results = run_baseline_tests(llm, baseline_df, housing_df, use_bedrock=args.bedrock)

    # Use LLM as judge to evaluate responses
    judge_responses_with_llm(llm, results)

    output_path = save_results(results)
    print_summary(results)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    llm_name = "Bedrock Llama 3.1 70B" if args.bedrock else "Gemini 2.5 Flash"
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Results saved to: {output_path}")
    print(f"LLM: {llm_name}")
    print(f"This baseline test used raw {llm_name} with dataset context but no agent system.")
    print(f"Compare these results with agent-based results to measure improvement.")


if __name__ == "__main__":
    main()