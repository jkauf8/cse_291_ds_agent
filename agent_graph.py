from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.planner import Planner
from agents.reviewer import Reviewer
from agents.final_reporter import FinalReporter
from tools.regression_tool import run_regression # Import the new tool
import pandas as pd # Import pandas
from tools.describe_data import describe_data 

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
    selected_tools: list  # Tools selected by planner (persists even when reviewer changes route)
    planner_iteration_count: int  # Track how many times planner has been called to prevent loops


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
        router_decision = route.get('router_decision', [])

        if "describe_data()" in router_decision and "run_regression()" in router_decision:
            return "DescribeAndRegress"
        elif "describe_data()" in router_decision:
            return 'DescribeData'
        elif "run_regression()" in router_decision:
            return 'RunRegression'
        elif "direct_response()" in router_decision:
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
        # Increment planner iteration counter
        current_iteration = state.get('planner_iteration_count', 0) + 1
        state['planner_iteration_count'] = current_iteration

        print(f"Planner: deciding tool to use... (iteration {current_iteration})")

        question = state.get('question', '')

        planner_result = self.planner.plan(question)

        selected_tools = planner_result.get("tools", [])
        state['route'] = {'router_decision': selected_tools}

        # Store the selected tools separately so they don't get overwritten by reviewer
        state['selected_tools'] = selected_tools

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

        # Check iteration count to prevent infinite loops
        max_iterations = 2
        current_iteration = state.get('planner_iteration_count', 0)

        # If we've already called the planner max_iterations times, force final report
        if current_iteration >= max_iterations:
            print(f"Reviewer: Maximum planner iterations ({max_iterations}) reached. Forcing final report.")
            state['route'] = {'router_decision': 'final_reporter'}
            print(f"Reviewer decision: {state['route']}")
            return state

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
        print("DescribeData: running dataset description...")

        question = state.get('question', '') or ''
        
        # Enhanced dataset selection with fallback
        selected_dataset_key = None
        q_lower = question.lower()
        
        # Try to match dataset from question
        for key in self.datasets.keys():
            if key.lower() in q_lower:
                selected_dataset_key = key
                break
        
        # Fallback to available datasets
        if selected_dataset_key is None:
            available_datasets = list(self.datasets.keys())
            if not available_datasets:
                state['tool_result'] = {"error": "No datasets available to describe."}
                return state
            selected_dataset_key = available_datasets[0]
            print(f"DescribeData: No specific dataset mentioned, using '{selected_dataset_key}'")

        df = self.datasets[selected_dataset_key]
        
        # Validate dataset is not empty
        if df.empty:
            state['tool_result'] = {"error": f"Dataset '{selected_dataset_key}' is empty after loading."}
            return state

        try:
            result = describe_data(df=df, dataset_name=selected_dataset_key)
        except Exception as e:
            import traceback
            traceback.print_exc()
            state['tool_result'] = {"error": f"DescribeData failed: {str(e)}"}
            return state

        # Attach structured result and metadata to state
        state['tool_result'] = result
        state['dataset_name'] = selected_dataset_key

        print("DescribeData: description complete.")
        return state

    def describe_and_regress_tool(self, state: State):
        """Runs describe_data and then run_regression"""
        print("Tool: Running Describe and then Regress...")
        # Run describe data first
        state = self.describe_data_tool(state)
        description_result = state['tool_result']

        # Then run regression
        state = self.run_regression_tool(state)
        regression_result = state['tool_result']
        
        # Combine results
        state['tool_result'] = [description_result] + regression_result
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
        workflow.add_node("describe_and_regress_tool", self.describe_and_regress_tool)
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
                "DescribeAndRegress": "describe_and_regress_tool",
                "DirectResponse": "report_agent",
            },
        )

        # Add edges from tools to review agent
        workflow.add_edge('describe_data_tool', "review_agent")
        workflow.add_edge('run_regression_tool', "report_agent") # Route directly to reporter
        workflow.add_edge('describe_and_regress_tool', "report_agent")


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
            'tool_result': None,
            'selected_tools': [],
            'planner_iteration_count': 0
        }

        result = self.app.invoke(initial_state)
        return result 