from .utils.messages import Assistant, Messages, System, User
from .logging import get_logger,LLMLogger, LogEntry
from .utils.retry import should_retry_exception
from .utils.tools import Tools, execute_tool, TEXT_EDITOR_TOOL
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from textwrap import dedent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from typing import Any, Optional, Callable
from dataclasses import dataclass
from functools import partial

import time
import uuid
import importlib


logger = get_logger(__name__)


def get_backend(name: str, async_api: bool = False):
    module_map = {
        "togetherai": "lmapis.providers.together",
        "together-ai": "lmapis.providers.together",
        "together": "lmapis.providers.together",
        "openai": "lmapis.providers.openai",
        "google": "lmapis.providers.google",
        "google-genai": "lmapis.providers.google_genai",
        "fireworks": "lmapis.providers.fireworks",
        "anthropic": "lmapis.providers.anthropic",
        "anthropic-bedrock": "lmapis.providers.anthropic_bedrock",
        "anthropic-vertex": "lmapis.providers.anthropic_vertex",
        "mistral": "lmapis.providers.mistral",
    }

    if name not in module_map:
        raise ValueError(f"Unexpected value {name}")

    module_path = module_map[name]
    module = importlib.import_module(module_path)

    return module.AsyncLMApi if async_api else module.LMApi


@dataclass
class Prompt:
    user: str
    system: str = None


def parse_finish_reason(output) -> str:
    try:
        finish_reason = output.choices[0].finish_reason
    except AttributeError:
        finish_reason = output.stop_reason
    
    return finish_reason


class LLM:
    def __init__(
        self,
        backend: str,
        model: str,
        cost: dict,
        credentials: str = None,
        model_params: dict = None,
        backend_kwargs: dict = None,
        logger: Optional[LLMLogger] = None,
    ):
        self.backend = backend
        self.model = model
        self.credentials = credentials
        self.cost = cost
        self.auto_cost_tracking = True
        self.model_params = model_params if model_params is not None else {}
        self.backend_kwargs = backend_kwargs if backend_kwargs is not None else {}
        self.llm_logger = logger
        self._call_cost = 0

        endpoint = get_backend(backend)
        self.llm = endpoint(api_key=self.credentials, **self.backend_kwargs)

    @property
    def is_prompt_caching_enabled(self):
        if "extra_headers" in self.model_params.keys():
            extras = self.model_params["extra_headers"]
            if extras.get("anthropic-beta") == "prompt-caching-2024-07-31":
                return True
        return False

    @property
    def call_cost(self) -> float:
        return self._call_cost

    def compute_cost(self, response: dict | Any | ChatCompletion) -> float:
        inputs = response.usage.prompt_tokens * float(self.cost["input"]) / 1_000_000
        outputs = response.usage.completion_tokens * \
                  float(self.cost["output"]) / 1_000_000
        return inputs + outputs

    def _log_interaction(
        self,
        messages: list,
        parameters: dict,
        response: Optional[ChatCompletion | ParsedChatCompletion],
        start_time: float,
        end_time: float,
        request_id: str,
        error: Optional[Exception] = None
    ) -> None:
        """Log a complete LLM interaction to the configured logger."""
        if not self.llm_logger or not self.llm_logger.is_enabled():
            return
        
        try:
            # Extract response data
            response_content = None
            finish_reason = None
            tokens_prompt = None
            tokens_completion = None
            cost = None
            
            if response is not None:
                if isinstance(response, ParsedChatCompletion):
                    try:
                        response_content = \
                            response.choices[0].message.parsed.model_dump(mode="json")
                    except Exception:   # noqa
                        logger.error("Failed to parse the model response to json")
                        response_content = ""
                    finish_reason = response.choices[0].finish_reason
                else:
                    # Extract response content
                    if hasattr(response, 'choices') and response.choices:
                        response_content = response.choices[0].message.content
                        finish_reason = response.choices[0].finish_reason
                    else:
                        response_content = getattr(response, 'content', str(response))
                        finish_reason = getattr(response, 'stop_reason', 'unknown')

                # Extract token usage and compute cost
                if hasattr(response, 'usage'):
                    tokens_prompt = response.usage.prompt_tokens
                    tokens_completion = response.usage.completion_tokens
                    cost = self.compute_cost(response)
            
            # Calculate duration
            duration_ms = int((end_time - start_time) * 1000)
            
            # Create and log entry
            log_entry = LogEntry.create(
                model=self.model,
                backend=self.backend,
                messages=messages,
                parameters=parameters,
                response_content=response_content,
                finish_reason=finish_reason,
                cost=cost,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                duration_ms=duration_ms,
                error=str(error) if error else None,
                retry_count=0,
                request_id=request_id,
            )
            
            self.llm_logger.log_interaction(log_entry)
            
        except Exception as log_error:
            logger.warning(f"Failed to log LLM interaction: {log_error}")

    def _get_messages(self, msgs: Messages, prefill_response: str | Assistant = None) -> Messages:
        if prefill_response is not None:
            prefill_response = Assistant(prefill_response) \
                if isinstance(prefill_response, str) else prefill_response

        # Prepare messages for the API call
        if "claude" in self.model:
            # Claude supports assistant prefill response
            if prefill_response is not None:
                # The prefill content cannot end with trailing whitespace. A prefill like
                # "As an AI assistant, I " (with a space at the end) will result in an error.
                prefill_response.content = prefill_response.content.rstrip()
            msgs = msgs >> prefill_response
        else:
            if prefill_response is not None:
                # For non-Claude models, add a user message to continue
                msgs = msgs >> prefill_response >> User(
                    content="Please continue where you left off. I will merge your "
                            "responses so make sure you don't add anything unrelated "
                            "in between"
                )
        return msgs

    def prepare_messages(
        self,
        *,
        prompt: Prompt = None,
        messages: Messages = None,
        assistant_prefill: str | Assistant = None,
    ) -> Messages:
        if messages is None:
            user = prompt.user
            system = prompt.system

            if system:
                system = dedent(system)

            messages = Messages() >> System(system) >> User(user)

        messages = self._get_messages(messages, assistant_prefill)
        return messages

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(10),
        retry=retry_if_exception(should_retry_exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.fn.__name__} "
            f"(attempt {retry_state.attempt_number}) due to: "
            f"{retry_state.outcome.exception()}"
        ),
        reraise=True
    )
    def _chat_completion(self, messages: Messages, **kwargs) -> ChatCompletion:
        return self.llm.client.chat.completions.create(
            model=self.model,
            messages=messages.get(),
            **self.model_params,
            **kwargs
        )

    def chat_completion(self, messages: Messages, **kwargs) -> ChatCompletion:
        # Generate unique request ID for this interaction
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            response = self._chat_completion(messages, **kwargs)
            end_time = time.time()
            
            self._log_interaction(
                messages=messages.get(),
                parameters={**self.model_params, **kwargs},
                response=response,
                start_time=start_time,
                end_time=end_time,
                request_id=request_id,
                error=None
            )
            
            return response
            
        except Exception as e:
            end_time = time.time()
            error = e
            
            self._log_interaction(
                messages=messages.get(),
                parameters={**self.model_params, **kwargs},
                response=None,
                start_time=start_time,
                end_time=end_time,
                request_id=request_id,
                error=error
            )
            
            # Re-raise the exception to maintain existing behavior
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(10),
        retry=retry_if_exception(should_retry_exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.fn.__name__} "
            f"(attempt {retry_state.attempt_number}) due to: "
            f"{retry_state.outcome.exception()}"
        ),
        reraise=True
    )
    def _chat_completion_parse(
        self, messages: Messages, response_format: Any, **kwargs
    ) -> Any:
        return self.llm.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages.get(),
            response_format=response_format,
            **self.model_params,
            **kwargs
        )

    def chat_completion_parse(
        self, messages: Messages, response_format: Any, **kwargs
    ) -> ChatCompletion:
        # Generate unique request ID for this interaction
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            response = self._chat_completion_parse(
                messages, response_format=response_format, **kwargs
            )
            end_time = time.time()

            self._log_interaction(
                messages=messages.get(),
                parameters={**self.model_params, **kwargs},
                response=response,
                start_time=start_time,
                end_time=end_time,
                request_id=request_id,
                error=None
            )

            return response

        except Exception as e:
            end_time = time.time()
            error = e

            self._log_interaction(
                messages=messages.get(),
                parameters={**self.model_params, **kwargs},
                response=None,
                start_time=start_time,
                end_time=end_time,
                request_id=request_id,
                error=error
            )

            # Re-raise the exception to maintain existing behavior
            raise

    @staticmethod
    def _extract_thinking_content(
        response: ChatCompletion | Any
    ) -> tuple[ChatCompletion | Any, str | None]:
        """
        Extract content between <think> tags if present and store it in reasoning_content.

        Args:
            response: The response object from the provider

        Returns:
            Modified response object
        """
        reasoning_content = None

        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                content = message.content.strip()
                if content.startswith("<think>") and "</think>" in content:
                    # Extract content between think tags
                    start_idx = len("<think>")
                    end_idx = content.find("</think>")
                    reasoning_content = content[start_idx:end_idx].strip()

                    # Remove the think tags from the original content
                    message.content = content[end_idx + len("</think>") :].strip()

        return response, reasoning_content

    def _call_model(
        self,
        *,
        prompt: Prompt = None,
        messages: Messages = None,
        assistant_prefill: str | Assistant = None,
        tools: Tools = None,
    ) -> Assistant:
        cost = 0

        if tools:
            tools = tools.format()

        extra_kwargs = dict(tools=tools) if tools is not None else {}

        messages = self.prepare_messages(
            prompt=prompt, messages=messages, assistant_prefill=assistant_prefill
        )
        output = self.chat_completion(messages, **extra_kwargs)
        output, reasoning_content = self._extract_thinking_content(output)
        finish_reason = parse_finish_reason(output)

        try:
            assistant = Assistant.from_model_response(output, reasoning_content)
        except BaseException as e:
            logger.error(
                f"Unable to parse assistant response, something is off. "
                f"Finish reason={finish_reason}"
            )
            logger.error(f"Model response={output.to_dict()}")
            raise e

        cost += self.compute_cost(output)

        # Iterate over stop reason if max tokens is reached and append output to input
        while finish_reason in ["max_tokens", "length", "model_length"]:
            # Reached max tokens, append output to input and reiterate
            logger.info(
                "Reached max token output, calling the model with prev output"
            )
            messages = self._get_messages(messages, assistant)
            output = self.chat_completion(messages, **extra_kwargs)
            output, reasoning_content = self._extract_thinking_content(output)
            new_response = Assistant.from_model_response(output, reasoning_content).content
            if new_response == "" or None:
                raise ValueError("Failed to generate a new response as model didn't "
                                 "complete the request within the defined output tokens")
            assistant.content += new_response
            # Update finish reason
            finish_reason = parse_finish_reason(output)
            cost += self.compute_cost(output)

        match finish_reason:
            case "end_turn" | "stop_sequence" | "stop" | "tool_calls" | "function_call":
                pass
            case _:
                # Model didn't stop naturally so we raise error
                logger.error(f"Model response={output.to_dict()}")
                raise ValueError("Model is finished with another reason.")

        self._call_cost = cost

        return assistant

    def __call__(
        self,
        *,
        prompt: Prompt = None,
        messages: Messages = None,
        assistant_prefill: str | Assistant = None,
        tools: Tools = None,
        max_turns: int = 0,
        **kwargs,
    ) -> Assistant:
        """
        Call an LLM endpoint.

        Args:
            prompt: Prompt that contains user and system text
            messages: Prompts in the defined `Messages` class
            assistant_prefill: Only for Claude models, response completion text
            tools: Specific tools for model to use.
            max_turns: If more than 1 then call will auto tool calling that many times.

        Returns:
            Assistant response
        """

        if max_turns == 0:
            assistant = self._call_model(
                prompt=prompt,
                messages=messages,
                assistant_prefill=assistant_prefill,
                tools=tools,
            )
            return assistant
        elif max_turns >= 1:
            turns = 0

            # We manage the whole messages history
            messages = self.prepare_messages(
                prompt=prompt, messages=messages, assistant_prefill=assistant_prefill
            )
            
            while turns <= max_turns:
                assistant = self._call_model(messages=messages, tools=tools)
            
                if not assistant.tool_calls:
                    break

                # First expand the messages with assistant response
                messages = messages >> assistant

                # Execute tools and get results
                # Use the current state of text files (output_text_files if available, otherwise input_text_files)
                current_text_files = messages.output_text_files \
                    if messages.output_text_files is not None else messages.input_text_files
                tool_messages = execute_tool(
                    tools, assistant.tool_calls, input_files=current_text_files
                )

                # Add tool messages to conversation
                for tm, atc in zip(tool_messages, assistant.tool_calls):
                    tool_name = atc.function.name
                    if tool_name == TEXT_EDITOR_TOOL["name"]:
                        # Tool call for text editor
                        if hasattr(tm, 'call_response') and tm.call_response is not None:
                            # Successful edit - use the updated text
                            messages.output_text_files = tm.call_response
                        else:
                            # View command or failed edit - preserve original text
                            if messages.output_text_files is None:
                                messages.output_text_files = messages.input_text_files
                    messages = messages >> tm

                turns += 1

            # Ensure the final assistant has the updated text files from the conversation
            assistant.text_files = messages.output_text_files

            return assistant
        else:
            raise ValueError("Expected max_turns to be positive integer")

    def _call_model_parse(
        self,
        *,
        response_format: Any,
        prompt: Prompt = None,
        messages: Messages = None,
        tools: Tools = None,
    ) -> Any:
        if tools:
            tools = tools.format()

        extra_kwargs = dict(tools=tools) if tools is not None else {}

        messages = self.prepare_messages(prompt=prompt, messages=messages)
        output = self.chat_completion_parse(
            messages, response_format=response_format, **extra_kwargs
        )
        finish_reason = parse_finish_reason(output)

        match finish_reason:
            case "end_turn" | "stop_sequence" | "stop":
                pass
            case _:
                # Model didn't stop naturally so we raise error
                try:
                    resp = output.to_dict()
                except AttributeError:
                    resp = output.model_dump(mode="python")

                logger.error(f"Model response={resp}")
                raise ValueError("Model is finished with another reason.")

        if output.choices[0].message.refusal is not None:
            # Model didn't respond naturally so we raise error
            logger.error(f"Model response={output.to_dict()}")
            raise ValueError("Model is finished with a refusal")

        cost = self.compute_cost(output)

        self._call_cost = cost

        return output.choices[0].message.parsed
