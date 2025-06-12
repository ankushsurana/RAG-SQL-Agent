from IPython.display import Image, display
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Represents the state structure containing a list of messages.

    Attributes:
        messages (list): A list of messages, where each message can be processed
        by adding messages using the `add_messages` function.
    """
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage.

    This class retrieves tool calls from the most recent AIMessage in the input
    and invokes the corresponding tool to generate responses.

    Attributes:
        tools_by_name (dict): A dictionary mapping tool names to tool instances.
    """

    def __init__(self, tools: list) -> None:
        """Initializes the BasicToolNode with available tools.
        """
        self.tools = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> State:
        """Executes the tools based on the tool calls in the last message.
        """
        message = state["messages"]
        last_message = message[-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool = self.tools[tool_call["name"]]
            content = tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=str(content),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages}

def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """Determines whether the chatbot should call a tool or respond directly.
    This function checks the last message in the state to see if it contains tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
        
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
    if isinstance(last_message, ToolMessage):
        return "chatbot"
            
    return "__end__"


def plot_agent_schema(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        return print("Graph could not be displayed.")
