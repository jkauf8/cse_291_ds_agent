from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.planner import Planner
from agents.reviewer import Reviewer
from agents.final_reporter import FinalReporter
from tools.describe_data import describe_data

class State(TypedDict):
    """State passed between nodes in the graph"""
    question: str
    response: str
    message_history: list
    route: dict
    tool_result: list


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

        selected_tool = self.planner.plan(question)

        state['route'] = {'router_decision': selected_tool}

        print(f"Planner decision: {state['route']}")

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
    def run_regression_tool(self, state: State):
        """Tool to run regression analysis"""
        # TODO: Implement actual regression logic
        state['tool_result'] = "Regression analysis: Random Forest model results"

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
        workflow.add_edge('run_regression_tool', "review_agent")

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