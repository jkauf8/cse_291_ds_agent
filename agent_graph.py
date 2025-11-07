from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.planner import Planner
from agents.reviewer import Reviewer
from agents.final_reporter import FinalReporter
from tools.regression_tool import run_regression # Import the new tool
import pandas as pd # Import pandas


class State(TypedDict):
    """State passed between nodes in the graph"""
    question: str
    response: str
    message_history: list
    route: dict
    tool_result: list
    dataset_name: str
    target_column: str
    feature_columns: list


class AgentGraph:
    """
    LangGraph-based agent workflow for data analysis.
    Orchestrates planner, tools, review, and reporting.
    """

    def __init__(self, llm, datasets: dict = None):
        """
        Initialize the agent graph.

        Args:
            llm: LLM instance
            datasets: Dictionary of loaded datasets (e.g., {'housing': df, 'coffee': df})
        """
        self.llm = llm
        self.datasets = datasets or {}
        self.planner = Planner(llm=llm)
        self.reviewer = Reviewer(llm=llm)
        self.final_reporter = FinalReporter(llm=llm)
        self.app = self.build_graph()

    def router(self, state: State):
        """Routes from planner to appropriate tool or directly to reporter"""
        route = state.get('route', {})
        router_decision = route.get('router_decision')

        if router_decision == 'describe_data()':
            return 'DescribeData'
        elif router_decision == 'run_regression()':
            return 'RunRegression'
        elif router_decision == 'direct_response()':
            return 'DirectResponse'
        else:
            return None

    def router2(self, state: State):
        """Routes from review agent back to planner or to final report"""
        route = state.get('route', {})
        router_decision = route.get('router_decision')

        if router_decision == 'planner':
            return 'Planner'
        elif router_decision == 'final_reporter':
            return 'FinalReporter'
        else:
            return None

    def planner_agent(self, state: State):
        """Planner agent - analyzes user question and selects appropriate tool"""
        print("Planner: deciding tool to use...")

        question = state.get('question', '')

        planner_result = self.planner.plan(question)
        
        selected_tool = planner_result.get("tool")
        state['route'] = {'router_decision': selected_tool}

        # Store planner results in state
        state['dataset_name'] = planner_result.get("dataset")
        state['target_column'] = planner_result.get("target_column")
        state['feature_columns'] = planner_result.get("feature_columns")


        print(f"Planner decision: {state['route']}")
        print(f"Dataset: {state['dataset_name']}, Target: {state['target_column']}, Features: {state['feature_columns']}")

        return state

    def review_agent(self, state: State):
        """Review agent - reviews tool results and decides next steps"""
        print("Reviewer: deciding to construct final report or call planner again...")

        question = state.get('question', '')
        tool_result = state.get('tool_result', [])

        routing_decision = self.reviewer.review(question, tool_result)

        state['route'] = {'router_decision': routing_decision}

        print(f"Reviewer decision: {state['route']}")

        return state

    def report_agent(self, state: State):
        """Report agent - generates final report in markdown format"""

        print("Reporter: constructing final report...")

        question = state.get('question', '')
        tool_result = state.get('tool_result', ['No results available'])

        # Use the FinalReporter to generate a comprehensive markdown report
        comprehensive_report = self.final_reporter.generate_report(question, tool_result)

        state['response'] = comprehensive_report
        state['message_history'] = state.get('message_history', [])

        state['message_history'].append(comprehensive_report)

        return state

    def describe_data_tool(self, state: State):
        """Tool to describe the dataset"""
        # TODO: Implement actual data description logic
        state['tool_result'] = "Data description: Statistical summary of the dataset"

        return state

    def run_regression_tool(self, state: State):
        """Tool to run regression analysis"""
        print("Tool: Running regression analysis...")
        
        dataset_name = state.get('dataset_name')
        target_column = state.get('target_column')
        feature_columns = state.get('feature_columns')

        if not dataset_name or not target_column:
            state['tool_result'] = ["Error: Dataset name or target column not provided by planner."]
            return state

        if dataset_name in self.datasets:
            try:
                df = self.datasets[dataset_name]

                # Ensure target column exists
                if target_column not in df.columns:
                    state['tool_result'] = [f"Error: Target column '{target_column}' not found in the '{dataset_name}' dataset."]
                    return state
                
                # If feature_columns are not specified, use all other columns
                if not feature_columns:
                    feature_columns = [col for col in df.columns if col != target_column]


                # Run the regression function from the tool
                results = run_regression(
                    df=df, 
                    target_column=target_column, 
                    feature_columns=feature_columns
                )
                
                # Store the summary in the state
                state['tool_result'] = [results['summary']]
                print(f"Tool: Regression complete. R2 score: {results['r2_score']:.4f}")

            except Exception as e:
                state['tool_result'] = [f"An error occurred during regression analysis: {str(e)}"]
                print(f"Error in run_regression_tool: {e}")
        else:
            state['tool_result'] = [f"Dataset '{dataset_name}' not available for regression."]

        return state

    def build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow"""
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("planner_agent", self.planner_agent)
        workflow.add_node("describe_data_tool", self.describe_data_tool)
        workflow.add_node("run_regression_tool", self.run_regression_tool)
        workflow.add_node("review_agent", self.review_agent)
        workflow.add_node("report_agent", self.report_agent)

        # Set entry point
        workflow.set_entry_point("planner_agent")

        # Add conditional edges from planner to tools or direct to reporter
        workflow.add_conditional_edges(
            "planner_agent",
            self.router,
            {
                "DescribeData": "describe_data_tool",
                "RunRegression": "run_regression_tool",
                "DirectResponse": "report_agent",
            },
        )

        # Add edges from tools to review agent
        workflow.add_edge('describe_data_tool', "review_agent")
        workflow.add_edge('run_regression_tool', "report_agent") # Route directly to reporter

        # Add conditional edges from review agent
        workflow.add_conditional_edges(
            "review_agent",
            self.router2,
            {
                "Planner": "planner_agent",
                "FinalReporter": "report_agent",
            },
        )

        # Add edge to end
        workflow.add_edge('report_agent', END)

        # Compile and return
        return workflow.compile()

    def run(self, question: str) -> dict:
        """
        Execute the agent graph on a user question.

        Args:
            question: User's research question

        Returns:
            Final state after execution
        """
        initial_state = {
            'question': question,
            'response': '',
            'message_history': [],
            'route': {},
            'tool_result': None
        }

        result = self.app.invoke(initial_state)
        return result 