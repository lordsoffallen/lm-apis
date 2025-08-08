from .utils.messages import Assistant, Messages, System, User, ChatCompletion
from .logging import get_logger,LLMLogger, LogEntry
from .utils.retry import should_retry_exception
from textwrap import dedent
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from typing import Any, Optional

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
        "mistral": "lmapis.providers.mistral",
    }

    if name not in module_map:
        raise ValueError(f"Unexpected value {name}")

    module_path = module_map[name]
    module = importlib.import_module(module_path)

    return module.AsyncLMApi if async_api else module.LMApi


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

        endpoint = get_backend(backend)
        self.llm = endpoint(api_key=self.credentials, **self.backend_kwargs)

    @property
    def is_prompt_caching_enabled(self):
        if "extra_headers" in self.model_params.keys():
            extras = self.model_params["extra_headers"]
            if extras.get("anthropic-beta") == "prompt-caching-2024-07-31":
                return True
        return False

    def compute_cost(self, response: dict | Any | ChatCompletion) -> float:
        inputs = response.usage.prompt_tokens * float(self.cost["input"]) / 1_000_000
        outputs = response.usage.completion_tokens * \
                  float(self.cost["output"]) / 1_000_000
        return inputs + outputs

    def _log_interaction(
        self,
        messages: list,
        parameters: dict,
        response: Optional[ChatCompletion],
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

    def _get_messages(self, messages: Messages, prefill_response: Assistant = None) -> Messages:
        # Prepare messages for the API call
        if "claude" in self.model:
            # Claude supports assistant prefill response
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
    def __llm_call(self, messages: Messages, **kwargs) -> ChatCompletion:
        return self.llm.client.chat.completions.create(
            model=self.model,
            messages=messages.get(),
            **self.model_params,
            **kwargs
        )

    def _call_model(
        self, msgs: Messages, prefill_response: Assistant = None, **kwargs
    ) -> ChatCompletion:
        # Generate unique request ID for this interaction
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        messages = self._get_messages(msgs, prefill_response)
        
        response = None
        error = None
        
        try:
            response = self.__llm_call(messages, **kwargs)
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

    def __call__(
        self,
        *,
        prompts: "Prompts" = None,
        messages: Messages = None,
        assistant_prefill: str | Assistant = None,
        tools: list[dict] = None,
        **kwargs,
    ) -> Assistant:
        """ Default chat completion endpoint """
        cost = 0

        if messages is None:
            query = prompts.user
            system_prompt = prompts.system

            if system_prompt:
                system_prompt = dedent(system_prompt)

            messages = Messages() >> System(system_prompt) >> User(query)

        if assistant_prefill is not None:
            assistant_prefill = Assistant(assistant_prefill) \
                if isinstance(assistant_prefill, str) else assistant_prefill

        output = self._call_model(messages, assistant_prefill, tools=tools)

        try:
            finish_reason = output.choices[0].finish_reason
        except AttributeError:
            finish_reason = output.stop_reason

        try:
            assistant = Assistant.from_model_response(output)
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

            output = self._call_model(messages, assistant, tools=tools)
            assistant.content += Assistant.from_model_response(output).content
            # Update finish reason
            try:
                finish_reason = output.choices[0].finish_reason
            except AttributeError:
                finish_reason = output.stop_reason
            cost += self.compute_cost(output)

        match finish_reason:
            case "end_turn" | "stop_sequence" | "stop" | "tool_calls" | "function_call":
                pass
            case _:
                # Model didn't stop naturally so we raise error
                logger.error(f"Model response={output.to_dict()}")
                raise ValueError("Model is finished with another reason.")

        return assistant
