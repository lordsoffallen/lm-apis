from typing import Any
from dataclasses import dataclass, field, asdict
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from copy import deepcopy

from .images import ImageContent


@dataclass
class TextContent:
    text: str
    type: str = field(default="text", init=False)


@dataclass
class RefusalContent:
    refusal: str
    type: str = field(default="refusal", init=False)


@dataclass
class BaseMessage:
    role: str
    content: str | list[TextContent | ImageContent | RefusalContent] | None


@dataclass
class System(BaseMessage):
    role: str = field(default="system", init=False)


@dataclass
class User(BaseMessage):
    role: str = field(default="user", init=False)


@dataclass
class Assistant(BaseMessage):
    role: str = field(default="assistant", init=False)
    reasoning_content: str = None
    function_call: Any = None   # deprecated
    tool_calls: Any | list[ChatCompletionMessageToolCall] = None
    refusal: str = None

    @classmethod
    def from_model_response(cls, output: ChatCompletion, reasoning_content: str = None) -> "Assistant":
        try:
            response = output.choices[0].message.to_dict()
        except AttributeError as e:
            response = output.choices[0].message.model_dump(mode="python")

        response.pop("role")  # Remove role as it's not required

        if "prefix" in response.keys():
            response.pop("prefix")

        tool_calls = response.get("tool_calls")
        if tool_calls is not None:
            # Parse into open ai type from dict here
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            tool_calls = [ChatCompletionMessageToolCall(**tc) for tc in tool_calls]
            response["tool_calls"] = tool_calls

        return cls(**response, reasoning_content=reasoning_content)

    def asdict(self) -> dict:
        d = asdict(self)
        d = {k:v for k, v in d.items() if v is not None}
        if "tool_calls" in d.keys():
            # Parse pydantic as dict here
            d["tool_calls"] = [i.model_dump(mode="python") for i in d["tool_calls"]]


@dataclass
class Tool(BaseMessage):
    role: str = field(default="tool", init=False)
    tool_call_id: str


@dataclass
class Function(BaseMessage):
    role: str = field(default="function", init=False)
    name: str


class Messages:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def add_message(self, message: BaseMessage = None) -> "Messages":
        new_copy = deepcopy(self)  # Create a copy first
        if message is not None:
            if (message.content is not None) and (message.content != ""):
                new_copy.messages.append(asdict(message))  # Modify the copy
        return new_copy  # Return the modified copy

    def __rshift__(self, other: BaseMessage = None) -> "Messages":
        """Implements the >> operator"""
        return self.add_message(other)

    def __str__(self) -> str:
        """String representation of the messages"""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.messages)

    def get(self):
        return self.messages
