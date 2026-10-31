import traceback
import uuid
from typing import List, Optional, Tuple
from chatbot.load_config import LoadProjectConfig
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory

PROJECT_CFG = LoadProjectConfig()

graph = build_graph()

create_directory(PROJECT_CFG.memory_dir)


class ChatBot:
    """
    A class to handle chatbot interactions by utilizing a pre-defined agent graph. The chatbot processes
    user messages, generates appropriate responses, and saves the chat history to a specified memory directory.
    """
    @staticmethod
    def respond(chatbot: List, message: str, thread_id: Optional[str]) -> Tuple:
        """
        Processes a user message using the agent graph, generates a response, and appends it to the chat history.
        The chat history is also saved to a memory file for future reference. A new conversation starts when
        there is no thread_id, and the thread_id to use for the next message is returned.
        """
        if not message.strip():
            return "", chatbot, thread_id

        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        try:
            events = graph.stream(
                {"messages": [{"role": "user", "content": message}]}, config, stream_mode="values"
            )
            for event in events:
                event["messages"][-1].pretty_print()
            response = event["messages"][-1].content
        except Exception as e:
            traceback.print_exc()
            response = f"Sorry, I couldn't answer that because of an error: {e}"

        chatbot.append((message, response))

        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot, folder_path=PROJECT_CFG.memory_dir, thread_id=thread_id)

        return "", chatbot, thread_id
