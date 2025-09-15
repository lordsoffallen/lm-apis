from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from copy import deepcopy

from .images import ImageContent

import warnings


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

    def __getattribute__(self, name):
        # Check if it's a dataclass field and marked as deprecated
        try:
            f = type(self).__dataclass_fields__.get(name)
            if f and f.metadata.get("deprecated"):
                warnings.warn(
                    f"Field '{name}' is deprecated and will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2
                )
        except AttributeError:
            pass
        return super().__getattribute__(name)

    def asdict(self) -> dict:
        """ Removes class attributes that starts with __ from the dict conversion """
        d = asdict(self)

        for k in d:
            if k.startswith("__"):
                d.pop(k)
        return d


@dataclass
class System(BaseMessage):
    role: str = field(default="system", init=False)


@dataclass
class User(BaseMessage):
    role: str = field(default="user", init=False)
    __file_inputs: str | list[str] = None

    @property
    def text_files(self):
        return self.__file_inputs

    @text_files.setter
    def text_files(self, value):
        self.__file_inputs = value


@dataclass
class Assistant(BaseMessage):
    role: str = field(default="assistant", init=False)
    reasoning_content: str = None
    function_call: Any = field(default=None, metadata={"deprecated": True})   # deprecated
    tool_calls: Any | list[ChatCompletionMessageToolCall] = None
    refusal: str = None
    __file_outputs: str | list[str] = None

    @property
    def text_files(self):
        return self.__file_outputs

    @text_files.setter
    def text_files(self, value):
        self.__file_outputs = value

    @classmethod
    def from_model_response(
        cls, output: ChatCompletion, reasoning_content: str = None
    ) -> "Assistant":
        try:
            response = output.choices[0].message.to_dict()
        except AttributeError:
            response = output.choices[0].message.model_dump(mode="python")

        tool_calls = response.get("tool_calls")
        if tool_calls is not None:
            # Parse into open ai type from dict here
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            tool_calls = [ChatCompletionMessageToolCall(**tc) for tc in tool_calls]
            response["tool_calls"] = tool_calls

        return cls(
            content=response.get("content"),
            refusal=response.get("refusal"),
            tool_calls=response.get("tool_calls"),
            reasoning_content=reasoning_content
        )

    def asdict(self) -> dict:
        d = asdict(self)
        d = {k:v for k, v in d.items() if v not in [None, []]}
        if "tool_calls" in d.keys():
            # Parse pydantic as dict here
            d["tool_calls"] = [i.model_dump(mode="python") for i in d["tool_calls"]]

        if "content" not in d:
            # content is none, add it back
            d["content"] = None

        return d


@dataclass
class Tool(BaseMessage):
    role: str = field(default="tool", init=False)
    tool_call_id: str
    __call_response: str = None

    @property
    def call_response(self):
        return self.__call_response

    @call_response.setter
    def call_response(self, value):
        self.__call_response = value


@dataclass
class Function(BaseMessage):
    role: str = field(default="function", init=False)
    name: str


class Messages:
    def __init__(self):
        self.messages: list[BaseMessage] = []
        self.__file_outputs: Optional[list[str] | str] = None
        self.__file_inputs: Optional[list[str] | str] = None

    @property
    def input_text_files(self):
        return self.__file_inputs

    @input_text_files.setter
    def input_text_files(self, value):
        self.__file_inputs = value

    @property
    def output_text_files(self):
        return self.__file_outputs

    @output_text_files.setter
    def output_text_files(self, value):
        self.__file_outputs = value

    def add_message(self, message: BaseMessage = None) -> "Messages":
        new_copy = deepcopy(self)  # Create a copy first
        if message is not None:
            # Add message if it has content OR if it has tool calls (for Assistant messages)
            has_content = (message.content is not None) and (message.content != "")
            has_tool_calls = (hasattr(message, 'tool_calls') and
                              message.tool_calls is not None and
                              len(message.tool_calls) > 0)
            
            if has_content or has_tool_calls:
                new_copy.messages.append(message)  # Store raw message object

                # Easy access for input/output files
                if getattr(message, "text_files", None):  # Safe access that returns None
                    if isinstance(message, Assistant):
                        new_copy.output_text_files = message.text_files
                    elif isinstance(message, User):
                        new_copy.input_text_files = message.text_files

        return new_copy  # Return the modified copy

    def __rshift__(self, other: BaseMessage = None) -> "Messages":
        """Implements the >> operator"""
        return self.add_message(other)

    def __str__(self) -> str:
        """String representation of the messages"""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)

    def get(self, as_dict: bool = True):
        """Get messages as dictionaries (default) or raw BaseMessage objects"""
        if as_dict:
            msgs = []
            for msg in self.messages:
                if hasattr(msg, "asdict"):
                    msgs.append(msg.asdict())
                else:
                    msgs.append(asdict(msg))
            return msgs
        return deepcopy(self.messages)
