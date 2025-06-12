from typing import List, Tuple
from chatbot.load_config import LoadProjectConfig
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory

PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

graph = build_graph()
config = {"configurable": {"thread_id": TOOLS_CFG.thread_id}}

create_directory("memory")


class ChatBot:
    """
    A class to handle chatbot interactions by utilizing a pre-defined agent graph. The chatbot processes
    user messages, generates appropriate responses, and saves the chat history to a specified memory directory.
    """
    @staticmethod
    def respond(chatbot: List, message: str) -> Tuple:
        """
        Processes a user message using the agent graph, generates a response, and appends it to the chat history.
        The chat history is also saved to a memory file for future reference.
        """
        events = graph.stream(
            {"messages": [{"role": "user", "content": message}]}, config, stream_mode="values"
        )
        for event in events:
            event["messages"][-1].pretty_print()

        chatbot.append(
            (message, event["messages"][-1].content))

        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot, folder_path=PROJECT_CFG.memory_dir, thread_id=TOOLS_CFG.thread_id)
        
        return "", chatbot
