# Data Science Agent: Tool-Augmented LLM for Automated Statistical Analysis

A multi-agent system that transforms natural language questions into rigorous statistical analysis through prompt engineering, tool augmentation, and architectural decoupling.

**Course**: CSE 291A - Generative AI | UC San Diego

## Overview

This project demonstrates how agent design patterns can transform a base LLM from **20% accuracy** to **90% accuracy** on data science tasks, while being **2.4x faster** than naive prompting approaches. The system uses a multi-agent LangGraph architecture where specialized agents (Planner, Reviewer, Reporter) orchestrate tool selection and analysis generation.

### Key Results

| Metric | Baseline LLM | Data Science Agent | Improvement |
|--------|--------------|-------------------|-------------|
| Ground Truth Accuracy | 20% (4/20) | 90% (18/20) | **4.5x more accurate** |
| Tool Selection Accuracy | N/A | 100% (20/20) | - |
| Avg. Time per Request | 47.38s | 19.63s | **2.4x faster** |

**Agent LLM**: Meta Llama 3.1 70B Instruct (AWS Bedrock)
**Evaluation Judge**: Gemini 2.5 Flash (consistent across all evaluations)

## Architecture

### Multi-Agent Design

![Agent Architecture](../architecture.png)

**Key Components**:
- **Planner** (`agents/planner.py`): Interprets queries and selects analytical tools
- **Reviewer** (`agents/reviewer.py`): Validates tool outputs and decides if re-planning is needed
- **Reporter** (`agents/final_reporter.py`): Synthesizes results into markdown reports
- **LLM-as-a-Judge**: Evaluates response quality against ground truth
- **Tools**:
  - `describe_data()`: Dataset statistics, distributions, correlations
  - `run_regression()`: Random Forest regression with feature importance

## Installation

### Prerequisites
- Python 3.11+
- AWS credentials (for Bedrock) AND Google API key (for Gemini)

### Setup

```bash
# Clone the repository
cd "/Users/yourname/CSE 291A/Project"

# Create virtual environment
python -m venv project_venv
source project_venv/bin/activate  # macOS/Linux
# or: project_venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `parent_dir/` directory:

```env
# For Gemini (alternative to Bedrock)
GEMINI_API_KEY=your_gemini_api_key_here

# For AWS Bedrock (primary LLM)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-west-1
```

## Usage

### Quick Interactive Mode (Gemini)

For quick experimentation with Gemini 2.5 Flash:

```bash
python run_agent.py
```

This launches an interactive session where you can ask questions and get real-time responses. Type `quit`, `exit`, or `q` to stop.

### Interactive Mode (Bedrock)

For interactive mode with Llama 3.1 70B (matching the evaluation setup):

```bash
python main.py --interact --bedrock
```

Example interaction:
```
Your question: What is the average price of houses?

RESPONSE:
The housing dataset contains 545 properties with an average price
of $540,296. Prices range from $75,000 to $13,300,000 with a
median of $450,000...
```

### Web Interface (Gradio)

Launch a browser-based chat interface:

```bash
python run_webapp.py          # Uses Gemini
python run_webapp.py --bedrock  # Uses Llama 3.1 70B
```

Opens a local web server with a chat UI for interacting with the agent.

### Batch Mode

Run predefined queries:
```bash
python main.py --bedrock
```

Results saved to `analysis_results.md`.

### Validation Testing

```bash
cd validation
python validation_test.py --bedrock  # Uses Llama 3.1 70B for agent, Gemini 2.5 Flash for judge
```

Evaluates agent on 20 test queries from `data/final_validation.csv` with automatic LLM-based judging. The `--bedrock` flag uses Llama 3.1 70B as the agent, while the judge is always Gemini 2.5 Flash for consistent evaluation.

### Baseline Comparison

```bash
cd baselines
python baseline1_test.py --bedrock  # Direct prompting baseline with Llama 3.1 70B, judged by Gemini 2.5 Flash
```

Runs the same 20 queries from `data/final_validation.csv` through a direct LLM approach (no agent system) for comparison.

**Final Evaluation Results**: The quantitative results reported in our final report used:
- Agent: `validation/validation_results_20251120_205139.csv`
- Baseline: `baselines/baseline_results_20251118_131340.csv`
- Both evaluated on `data/final_validation.csv` (20 queries with ground truth)

## Key Features

### 1. Tool-Augmented Reasoning
The agent doesn't generate Python code or hallucinate statistics. It selects from pre-validated analytical tools:

```python
# Planner selects tool based on query
{
  "tools": ["describe_data()"],
  "dataset": "housing",
  "target_column": null
}

# Tool executes with actual data
result = describe_data(df=housing_df, dataset_name="housing")
```

### 2. Multi-Stage Review Process
The Reviewer agent validates tool outputs:
- Did the tool selection make sense?
- Is additional analysis needed?
- Should we iterate back to the Planner?

This prevents premature conclusions and ensures comprehensive analysis.

### 3. Prompt Engineering
Specialized prompts for each agent:
- **Planner**: Few-shot examples demonstrating tool selection patterns
- **Reviewer**: Evaluation criteria for output quality
- **Reporter**: Markdown formatting and explanation guidelines

### 4. LLM-as-a-Judge Evaluation
Automated validation using **Gemini 2.5 Flash** as the judge (consistent across both baseline and agent evaluations):
```python
judge_prompt = """
Compare the agent's answer to the ground truth.
Score 1 if semantically equivalent, 0 otherwise.
"""
```

**Evaluation Framework**:
- **Agent/Baseline LLM**: Llama 3.1 70B (performs the actual data analysis)
- **Judge LLM**: Gemini 2.5 Flash (evaluates output quality)
- **Rationale**: Keeping the judge separate ensures consistent evaluation methodology across all approaches and prevents potential biases from self-evaluation

## Evaluation Metrics

### Tool Selection Accuracy (TSA)
Correctness of tool choice for a given query:
- **Current**: 100% (20/20)

### Ground Truth Accuracy (GTA)
Semantic correctness of final answers via LLM judge:
- **Current**: 90% (18/20)

### Per-Tool Breakdown
- `describe_data()`: 90% accuracy (9/10)
- `run_regression()`: 75% accuracy (6/8)
- Combined usage: 100% accuracy (2/2)

## Comparison to Baselines

| Approach | Description | Accuracy | Avg. Time |
|----------|-------------|----------|-----------|
| **Baseline** | Direct prompting | 20% | 47.38s |
| **This Project** | Tool-augmented agent | **90%** | **19.63s** |

The agent achieves higher accuracy AND faster execution by:
1. Offloading computation to deterministic tools (no "reasoning through" math)
2. Shorter inference chains (tool selection is simpler than full analysis)
3. Focused responses (tools return structured data, not verbose explanations)


## Datasets Utilized

### Housing Dataset
- **Size**: 545 properties × 13 features
- **Target**: `price` (continuous)
- **Features**: Area, bedrooms, bathrooms, stories, parking, etc.
- **Use Case**: Predicting house prices based on property characteristics


## LLM Usage
LLMs were utilized in the general-purpose assistance of this project and code development. LLM was used to explore conceptual ideas, provide guidance, and with assistance in code development and debugging. All LLM-generated content was reviewed and modified for understanding and accuracy.