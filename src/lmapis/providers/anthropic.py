from lmapis.base import BaseLMApi
from lmapis.utils import get_api_key_from_env
from anthropic import Anthropic
from anthropic.resources import messages as utils   # refer module functions
from anthropic import NOT_GIVEN, NotGiven
from typing import Optional, Literal, Iterable, Any, Dict, Union, List
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat import ChatCompletionMessage
from openai.types import CompletionUsage
from time import time


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


class CompletionsAnthropic(utils.Messages):
    @staticmethod
    def warn_for_non_supported_params(*args):
        for arg in args:
            if arg != NOT_GIVEN:
                raise ValueError(f"{arg} is not supported in Anthropic")

    def create(
        self,
        messages: Iterable[utils.MessageParam],
        model: utils.ModelParam,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: NotGiven = NOT_GIVEN,
        functions: Iterable[Any] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: int | Literal[4096] | Literal[8192] = 8192,
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
        tools: Iterable[utils.ToolParam] | NotGiven = NOT_GIVEN,
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

        if not model.startswith("claude-3-5"):
            max_tokens = 4096   # Other models support less max tokens

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

        system = []

        for m in messages[:]:   # remove breaks the loop in vanilla iter so we copy here
            if m["role"] == "system":
                # extract system prompt from the messages
                system.append({"type": "text", "text": m["content"]})
                messages.remove(m)      # noqa

        if len(system) == 0:
            system = NOT_GIVEN

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

        return ChatCompletion(
            id=response.id,
            choices=[
                Choice(
                    finish_reason=reason_mapping[response.stop_reason],
                    index=0,
                    message=ChatCompletionMessage(
                        # We assume it's always text type for now.
                        content=response.content[0].text,
                        role=response.role,
                        # TODO may add support later for tool calling
                        # refusal=,
                        # function_call,
                        # tool_calls,
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
