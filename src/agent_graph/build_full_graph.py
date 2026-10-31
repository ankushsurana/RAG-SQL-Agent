from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_core.messages import trim_messages
from agent_graph.tool_chinook_sqlagent import query_chinook_sqldb
from agent_graph.tool_travel_sqlagent import query_travel_sqldb
from agent_graph.tool_lookup_policy_rag import lookup_swiss_airline_policy
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.tool_novel_rag import lookup_novel
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.agent_backend import State, BasicToolNode, route_tools
from agent_graph.llm_utils import load_llm

TOOLS_CFG = LoadToolsConfig()


def build_graph():
    """
    Builds an agent decision-making graph by combining an LLM with various tools
    and defining the flow of interactions between them.
    """
    memory = MemorySaver()
    primary_llm = load_llm(
        repo_id=TOOLS_CFG.primary_agent_llm,
        temperature=TOOLS_CFG.primary_agent_llm_temperature
    )

    graph_builder = StateGraph(State)

    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_swiss_airline_policy,
             lookup_novel,
             query_travel_sqldb,
             query_chinook_sqldb,
             ]

    llm_with_tools = primary_llm.bind_tools(tools)

    def chatbot(state: State):
        """Executes the primary language model with tools bound and returns the generated message."""
        # Starting on a human message keeps tool results together with the tool calls they answer.
        messages = trim_messages(
            state["messages"],
            max_tokens=TOOLS_CFG.primary_agent_max_history_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph_builder.add_node("chatbot", chatbot)
    tool_node = BasicToolNode(tools=tools)

    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")

    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    graph_builder.add_edge("tools", "chatbot")

    graph = graph_builder.compile(checkpointer=memory)
    return graph
