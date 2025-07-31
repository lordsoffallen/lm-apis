from lmapis.base import BaseLMApi, BaseAsyncLMApi
from lmapis.utils import get_api_key_from_env
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock, ToolUseBlockParam, ToolResultBlockParam
from anthropic.resources import messages as utils   # refer module functions
from anthropic import NOT_GIVEN, NotGiven
from typing import Optional, Literal, Iterable, Any, Dict, Union, List
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam, \
    ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall, Function
from openai.types import CompletionUsage
from time import time

import json


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("ANTHROPIC_API_KEY")

        super().__init__(
            base_url="https://api.anthropic.com", api_key=api_key, **kwargs
        )

    @property
    def client(self):
        if self._client is None:
            self._client = LMApiAnthropic(api_key=self.api_key, **self.kwargs)
        return self._client


def _convert_single_message(
    msg: ChatCompletionMessageParam | dict
) -> utils.MessageParam | dict:
    # TODO image converting is not supported yet
    if not isinstance(msg["content"], str):
        try:
            if msg["content"].get("type") == "image_url":
                raise NotImplementedError("Image parsing is not implemented yet")
        except AttributeError as e:
            # get method does not exist, ideally not image content
            pass

    if msg["role"] == "tool":
        return utils.MessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=msg["tool_call_id"],
                    content=msg["content"],
                    # Infer if message contains error message or not
                    is_error=True if "error" in msg["content"].lower() else False
                )
            ]
        )
    elif msg["role"] == "assistant" and "tool_calls" in msg:
        message_content = []
        if msg["content"]:
            message_content.append(TextBlock(type="text", text=msg["content"]))

        for tool_call in msg["tool_calls"]:
            tool_input = (
                tool_call["function"]["arguments"]
                if isinstance(tool_call, dict)
                else tool_call.function.arguments
            )
            message_content.append(
                ToolUseBlockParam(
                    type="tool_use",
                    input=json.loads(tool_input),
                    id=tool_call["id"] if isinstance(tool_call, dict) else tool_call.id,
                    name=tool_call["function"]["name"]
                        if isinstance(tool_call, dict)
                        else tool_call.function.name
                )
            )
        return utils.MessageParam(
            role="assistant", content=message_content
        )
    return utils.MessageParam(
        role= msg["role"], content=msg["content"]
    )


def convert_messages(
    messages: Iterable[ChatCompletionMessageParam]
) -> tuple[list[TextBlock], Iterable[utils.MessageParam]]:
    system = []

    for m in messages[:]:  # remove breaks the loop in vanilla iter so we copy here
        if m["role"] == "system":
            # extract system prompt from the messages
            system.append({"type": "text", "text": m["content"]})
            messages.remove(m)  # noqa

    if len(system) == 0:
        system = NOT_GIVEN

    converted_messages = [_convert_single_message(msg) for msg in messages]

    return system, converted_messages


def convert_tools(
    tools: Iterable[ChatCompletionToolParam] | dict | None
) -> Iterable[utils.ToolParam] | None:
    anthropic_tools = []

    if tools is None or tools == NOT_GIVEN:
        return NOT_GIVEN

    for tool in tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        anthropic_tool = utils.ToolParam(
            name=function["name"],
            description= function["description"],
            input_schema={
                "type": "object",
                "properties": function["parameters"]["properties"],
                "required": function["parameters"].get("required", []),
            }
        )
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


class CompletionsAnthropic(utils.Messages):
    @staticmethod
    def warn_for_non_supported_params(*args):
        for arg in args:
            if arg != NOT_GIVEN:
                raise ValueError(f"{arg} is not supported in Anthropic")

    def create(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: utils.ModelParam,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: NotGiven = NOT_GIVEN,
        functions: Iterable[Any] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: int = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: NotGiven = NOT_GIVEN,
        # Anthropic uses stop_sequences so we map it to stop here
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        stream_options: NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: utils.message_create_params.ToolChoice | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] |  NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,  # ANTHROPIC SPECIFIC PARAM
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: utils.Headers | None = None,
        extra_query: utils.Query | None = None,
        extra_body: utils.Body | None = None,
        timeout: float | utils.httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        self.warn_for_non_supported_params(
            frequency_penalty,
            function_call,
            functions,
            logit_bias,
            logprobs,
            n,
            stream_options,
            parallel_tool_calls,
            presence_penalty,
            response_format,
            seed,
            service_tier,
            top_logprobs,
            user
        )

        if max_tokens == NOT_GIVEN:
            # Auto infer max tokens as it is a required arg for claude
            max_tokens = 8192 if model.startswith("claude-3-5") else 4096

        if (not utils.is_given(timeout) and
            self._client.timeout == utils.DEFAULT_TIMEOUT):
            timeout = 600

        if model in utils.DEPRECATED_MODELS:
            utils.warnings.warn(
                f"The model '{model}' is deprecated and will reach end-of-life on "
                f"{utils.DEPRECATED_MODELS[model]}.\nPlease migrate to a newer model. "
                f"Visit https://docs.anthropic.com/en/docs/resources/model-deprecations"
                f" for more information.",
                DeprecationWarning,
                stacklevel=3,
            )

        tools = convert_tools(tools)
        system, messages = convert_messages(messages)

        response = self._post(
            "/v1/messages",
            body=utils.maybe_transform(
                {
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "model": model,
                    "metadata": NOT_GIVEN,
                    "stop_sequences": stop,
                    "stream": stream,
                    "system": system,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                utils.message_create_params.MessageCreateParams,
            ),
            options=utils.make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout
            ),
            cast_to=utils.Message,
            stream=stream or False,     # noqa
            stream_cls=utils.Stream[utils.RawMessageStreamEvent],
        )

        if not isinstance(response,  utils.Message):
            # Support non-streaming for now
            raise ValueError(f"Got unexpected response type: {type(response)}")

        # Map Message to ChatCompletion to have seamless parsing experience
        # Only support non-streaming for now.

        reason_mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }

        text_content = None
        tool_calls = []

        for r in response.content:
            if isinstance(r, TextBlock):
                text_content = r.text
            if isinstance(r, ToolUseBlock):
                tool_content = ChatCompletionMessageToolCall(
                    id=r.id,
                    function=Function(
                        name=r.name,
                        arguments=json.dumps(r.input),
                    ),
                    type="function"
                )
                tool_calls.append(tool_content)

        return ChatCompletion(
            id=response.id,
            choices=[
                Choice(
                    finish_reason=reason_mapping[response.stop_reason],
                    index=0,
                    message=ChatCompletionMessage(
                        content=text_content,
                        role=response.role,
                        tool_calls=tool_calls
                        # refusal=,
                    ),
                )
            ],
            created=int(time()),
            model=response.model,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=response.usage.output_tokens,
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.output_tokens + response.usage.input_tokens
            ),
        )


class ChatAnthropic:
    def __init__(self, client):
        self._client = client

    @utils.cached_property
    def completions(self):
        return CompletionsAnthropic(self._client)


class LMApiAnthropic(Anthropic):
    @utils.cached_property
    def chat(self):
        return ChatAnthropic(self)


class AsyncLMApi(BaseAsyncLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("ANTHROPIC_API_KEY")

        super().__init__(
            base_url="https://api.anthropic.com", api_key=api_key, **kwargs
        )

    @property
    def client(self):
        if self._client is None:
            self._client = AsyncLMApiAnthropic(api_key=self.api_key, **self.kwargs)
        return self._client


class AsyncCompletionsAnthropic(utils.AsyncMessages):
    @staticmethod
    def warn_for_non_supported_params(*args):
        for arg in args:
            if arg != NOT_GIVEN:
                raise ValueError(f"{arg} is not supported in Anthropic")

    async def create(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: utils.ModelParam,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: NotGiven = NOT_GIVEN,
        functions: Iterable[Any] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: int = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: NotGiven = NOT_GIVEN,
        # Anthropic uses stop_sequences so we map it to stop here
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        stream_options: NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: utils.message_create_params.ToolChoice | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] |  NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,  # ANTHROPIC SPECIFIC PARAM
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: utils.Headers | None = None,
        extra_query: utils.Query | None = None,
        extra_body: utils.Body | None = None,
        timeout: float | utils.httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        self.warn_for_non_supported_params(
            frequency_penalty,
            function_call,
            functions,
            logit_bias,
            logprobs,
            n,
            stream_options,
            parallel_tool_calls,
            presence_penalty,
            response_format,
            seed,
            service_tier,
            top_logprobs,
            user
        )

        if max_tokens == NOT_GIVEN:
            # Auto infer max tokens as it is a required arg for claude
            max_tokens = 8192 if model.startswith("claude-3-5") else 4096

        if (not utils.is_given(timeout) and
            self._client.timeout == utils.DEFAULT_TIMEOUT):
            timeout = 600

        if model in utils.DEPRECATED_MODELS:
            utils.warnings.warn(
                f"The model '{model}' is deprecated and will reach end-of-life on "
                f"{utils.DEPRECATED_MODELS[model]}.\nPlease migrate to a newer model. "
                f"Visit https://docs.anthropic.com/en/docs/resources/model-deprecations"
                f" for more information.",
                DeprecationWarning,
                stacklevel=3,
            )

        tools = convert_tools(tools)
        system, messages = convert_messages(messages)

        response = await self._post(
            "/v1/messages",
            body=await utils.async_maybe_transform(
                {
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "model": model,
                    "metadata": NOT_GIVEN,
                    "stop_sequences": stop,
                    "stream": stream,
                    "system": system,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                utils.message_create_params.MessageCreateParams,
            ),
            options=utils.make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout
            ),
            cast_to=utils.Message,
            stream=stream or False,     # noqa
            stream_cls=utils.AsyncStream[utils.RawMessageStreamEvent],
        )

        if not isinstance(response,  utils.Message):
            # Support non-streaming for now
            raise ValueError(f"Got unexpected response type: {type(response)}")

        # Map Message to ChatCompletion to have seamless parsing experience
        # Only support non-streaming for now.

        reason_mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }

        text_content = None
        tool_calls = []

        for r in response.content:
            if isinstance(r, TextBlock):
                text_content = r.text
            if isinstance(r, ToolUseBlock):
                tool_content = ChatCompletionMessageToolCall(
                    id=r.id,
                    function=Function(
                        name=r.name,
                        arguments=json.dumps(r.input),
                    ),
                    type="function"
                )
                tool_calls.append(tool_content)

        return ChatCompletion(
            id=response.id,
            choices=[
                Choice(
                    finish_reason=reason_mapping[response.stop_reason],
                    index=0,
                    message=ChatCompletionMessage(
                        content=text_content,
                        role=response.role,
                        tool_calls=tool_calls
                        # refusal=,
                    ),
                )
            ],
            created=int(time()),
            model=response.model,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=response.usage.output_tokens,
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.output_tokens + response.usage.input_tokens
            ),
        )


class AsyncChatAnthropic:
    def __init__(self, client):
        self._client = client

    @utils.cached_property
    def completions(self):
        return AsyncCompletionsAnthropic(self._client)


class AsyncLMApiAnthropic(AsyncAnthropic):
    @utils.cached_property
    def chat(self):
        return AsyncChatAnthropic(self)
