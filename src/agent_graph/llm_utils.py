import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.chat_models.huggingface import _convert_message_to_chat_message

MAX_NEW_TOKENS = 1000


class ToolCallingChatHuggingFace(ChatHuggingFace):
    """
    ChatHuggingFace with the chat completion calls corrected for tool use. The base class in
    langchain-huggingface 0.0.3:
    - ignores temperature, max_new_tokens and stop.
    - replaces every ' with " in the tool arguments, which loses any tool call whose query has an apostrophe.
    - sends tool calls and tool results without their ids, which the OpenAI-style providers need.
    """

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        """Calls the chat completion API with the configured parameters and converts the response.
        """
        kwargs.setdefault("temperature", self.llm.temperature)
        kwargs.setdefault("max_tokens", self.llm.max_new_tokens)
        if stop:
            kwargs["stop"] = stop
        response = self.llm.client.chat_completion(
            messages=self._create_message_dicts(messages, stop), **kwargs)
        choice = response.choices[0]
        generation = ChatGeneration(
            message=self._to_ai_message(choice.message),
            generation_info={"finish_reason": choice.finish_reason},
        )
        return ChatResult(generations=[generation], llm_output={"token_usage": response.usage})

    def _create_message_dicts(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> List[Dict]:
        """Converts the messages to the chat completion format, keeping the tool call ids.
        """
        return [self._to_message_dict(message) for message in messages]

    @staticmethod
    def _to_message_dict(message: BaseMessage) -> Dict:
        if isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["args"]),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            return message_dict
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "content": message.content,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
            }
        return _convert_message_to_chat_message(message)

    @staticmethod
    def _to_ai_message(message: Any) -> AIMessage:
        additional_kwargs = {}
        if message.tool_calls:
            additional_kwargs["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        # Some providers return the arguments as an object instead of a JSON string.
                        "arguments": tool_call.function.arguments
                        if isinstance(tool_call.function.arguments, str)
                        else json.dumps(tool_call.function.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return AIMessage(content=message.content or "", additional_kwargs=additional_kwargs)


def load_llm(repo_id: str, temperature: float) -> ToolCallingChatHuggingFace:
    """
    Creates the chat model for the given Hugging Face repo id.
    """
    return ToolCallingChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            max_new_tokens=MAX_NEW_TOKENS
        )
    )
