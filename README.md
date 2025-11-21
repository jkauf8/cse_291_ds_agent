# Data Science Agent: Tool-Augmented LLM for Automated Statistical Analysis

A multi-agent system that transforms natural language questions into rigorous statistical analysis through prompt engineering, tool augmentation, and architectural decoupling.

**Course**: CSE 291A - Generative AI | UC San Diego

## Overview

This project demonstrates how agent design patterns can transform a base LLM from **21% accuracy** to **89.5% accuracy** on data science tasks, while being **2.4x faster** than naive prompting approaches. The system uses a multi-agent LangGraph architecture where specialized agents (Planner, Reviewer, Reporter) orchestrate tool selection and analysis generation.

### Key Results

| Metric | Baseline LLM | Data Science Agent | Improvement |
|--------|--------------|-------------------|-------------|
| Ground Truth Accuracy | 21.1% (4/19) | 89.5% (17/19) | **+324%** |
| Tool Selection Accuracy | N/A | 100% (20/20) | - |
| Avg. Time per Request | 47.38s | 19.63s | **2.4x faster** |

**LLM Backend**: Meta Llama 3.1 70B Instruct (AWS Bedrock)

See [`qualitative_results_comparison.md`](qualitative_results_comparison.md) for detailed analysis.

## Architecture

### Multi-Agent Design

```
User Question
     ↓
[Planner Agent] ──→ Tool Selection (describe_data, run_regression, both)
     ↓
[Tool Execution] ──→ Statistical Computation
     ↓
[Reviewer Agent] ──→ Quality Check (iterate or proceed?)
     ↓
[Reporter Agent] ──→ Natural Language Report
```

**Key Components**:
- **Planner** (`agents/planner.py`): Interprets queries and selects analytical tools
- **Reviewer** (`agents/reviewer.py`): Validates tool outputs and decides if re-planning is needed
- **Reporter** (`agents/final_reporter.py`): Synthesizes results into markdown reports
- **Tools**:
  - `describe_data()`: Dataset statistics, distributions, correlations
  - `run_regression()`: Random Forest regression with feature importance

### Design Philosophy

Unlike systems that generate arbitrary Python code (AI Scientist) or use complex multi-agent hierarchies (6+ agents), this project uses:
- **Constrained toolset**: Only 2 predefined analytical functions
- **Linear reasoning path**: Predictable describe → analyze → report workflow
- **Deterministic behavior**: Same query produces same tool selection
- **Interpretability**: Full visibility into agent reasoning and tool choices

## Project Structure

```
Project/
├── code/
│   ├── main.py                    # Entry point with interactive mode
│   ├── agent_graph.py             # LangGraph workflow orchestration
│   ├── gemini_llm.py              # Gemini LLM integration
│   ├── agents/
│   │   ├── planner.py             # Tool selection agent
│   │   ├── reviewer.py            # Quality assurance agent
│   │   └── final_reporter.py     # Report generation agent
│   ├── tools/
│   │   ├── describe_data.py       # Statistical description tool
│   │   └── regression_tool.py    # Random Forest regression tool
│   ├── prompts/
│   │   ├── planner_prompt.py      # Few-shot tool selection prompts
│   │   ├── reviewer_prompt.py     # Review criteria prompts
│   │   └── final_reporter_prompt.py  # Report formatting prompts
│   ├── data/
│   │   ├── housing.csv            # Housing dataset (545 × 13)
│   │   └── coffee_shop_sales.xlsx # Coffee sales dataset (149K × 14) (not utilized)
│   ├── validation/
│   │   ├── validation.csv         # Test queries with ground truth
│   │   └── validation_test.py     # Automated evaluation script
│   └── baselines/
│       ├── baseline1_test.py      # Direct prompting baseline
│       └── baseline2_test.py      # Chain-of-thought baseline
├── qualitative_results_comparison.md  # Performance analysis
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.11+
- AWS credentials (for Bedrock) OR Google API key (for Gemini)

### Setup

```bash
# Clone the repository
cd "/Users/yourname/CSE 291A/Project"

# Create virtual environment
python -m venv project_venv
source project_venv/bin/activate  # macOS/Linux
# or: project_venv\Scripts\activate  # Windows

# Install dependencies
cd code
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `code/` directory:

```env
# For Gemini (alternative to Bedrock)
GEMINI_API_KEY=your_gemini_api_key_here

# For AWS Bedrock (primary LLM)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-west-1
```

## Usage

### Interactive Mode

```bash
cd code
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

### Batch Mode

Run predefined queries:
```bash
python main.py --bedrock
```

Results saved to `analysis_results.md`.

### Validation Testing

```bash
cd validation
python validation_test.py
```

Evaluates agent on 20 test queries with automatic LLM-based judging.

### Baseline Comparison

```bash
cd baselines
python baseline1_test.py  # Direct prompting baseline
```

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
Automated validation using the same LLM (Llama 3.1 70B) as a judge:
```python
judge_prompt = """
Compare the agent's answer to the ground truth.
Score 1 if semantically equivalent, 0 otherwise.
"""
```

## Evaluation Metrics

### Tool Selection Accuracy (TSA)
Correctness of tool choice for a given query:
- **Current**: 100% (20/20)

### Ground Truth Accuracy (GTA)
Semantic correctness of final answers via LLM judge:
- **Current**: 89.5% (17/19)

### Per-Tool Breakdown
- `describe_data()`: 90% accuracy (9/10)
- `run_regression()`: 75% accuracy (6/8)
- Combined usage: 100% accuracy (2/2)

## Comparison to Baselines

| Approach | Description | Accuracy | Avg. Time |
|----------|-------------|----------|-----------|
| **Baseline** | Direct prompting | 21.1% | 47.38s |
| **This Project** | Tool-augmented agent | **89.5%** | **19.63s** |

The agent achieves higher accuracy AND faster execution by:
1. Offloading computation to deterministic tools (no "reasoning through" math)
2. Shorter inference chains (tool selection is simpler than full analysis)
3. Focused responses (tools return structured data, not verbose explanations)

## Technical Details

### LangGraph State Management
The agent graph uses typed state to pass information between nodes:

```python
class State(TypedDict):
    question: str
    route: dict  # Tool selection routing
    tool_result: list  # Outputs from tools
    dataset_name: str
    target_column: str
    selected_tools: list
    planner_iteration_count: int
```

### Conditional Routing
The graph supports dynamic workflows:

```python
workflow.add_conditional_edges(
    "planner_agent",
    self.router,
    {
        "DescribeData": "describe_data_tool",
        "RunRegression": "run_regression_tool",
        "DescribeAndRegress": "describe_and_regress_tool",
    },
)
```

### Loop Prevention
Maximum 2 Planner iterations to prevent infinite loops:
```python
if current_iteration >= max_iterations:
    print("Forcing final report")
    state['route'] = {'router_decision': 'final_reporter'}
```

## Datasets

### Housing Dataset
- **Size**: 545 properties × 13 features
- **Target**: `price` (continuous)
- **Features**: Area, bedrooms, bathrooms, stories, parking, etc.
- **Use Case**: Predicting house prices based on property characteristics