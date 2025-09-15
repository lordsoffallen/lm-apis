import pytest
import os
import json

from unittest.mock import Mock, patch
from lmapis.llm import LLM, Prompt, parse_finish_reason, get_backend
from lmapis.logging import LLMLogger, LogEntry
from lmapis.utils.messages import Messages, System, User, Assistant
from lmapis.utils.tools import Tools, TEXT_EDITOR_TOOL
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall, Function


class TestLLMMaxTurnsAndToolCalling:
    """Test suite for LLM max_turns parameter and tool calling functionality."""
    
    @pytest.fixture
    def mock_backend_setup(self):
        """Setup mock backend for tool calling tests."""
        mock_client = Mock()
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_instance.client = mock_client
        mock_backend_class.return_value = mock_backend_instance
        return mock_backend_class, mock_client
    
    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b
        
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 25°C"

        return Tools([add_numbers, get_weather])

    @staticmethod
    def create_mock_response_with_tool_calls(
        tool_calls_data=None, content="I'll help you with that."
    ):
        """Create a mock response with tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        
        if tool_calls_data:
            tool_calls = []
            for i, (name, args) in enumerate(tool_calls_data):
                tool_call = ChatCompletionMessageToolCall(
                    id=f"call_{i}",
                    function=Function(name=name, arguments=json.dumps(args)),
                    type="function"
                )
                tool_calls.append(tool_call)
            
            mock_response.choices[0].message.tool_calls = tool_calls
        else:
            mock_response.choices[0].message.tool_calls = None
        
        return mock_response

    @staticmethod
    def create_mock_final_response(content="Task completed successfully."):
        """Create a mock final response without tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 15
        return mock_response
    
    def test_max_turns_zero_no_tool_calling(self, mock_backend_setup, sample_tools):
        """Test that max_turns=0 works like normal call without tool execution."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Mock response with tool calls but max_turns=0 should not execute them
        mock_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": 5, "b": 3})
        ])
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            mock_assistant = Assistant("I'll help you with that.")
            mock_assistant.tool_calls = mock_response.choices[0].message.tool_calls
            mock_from_response.return_value = mock_assistant
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Add 5 and 3")
            result = llm(messages=messages, tools=sample_tools, max_turns=0)
            
            # Should only make one API call
            assert mock_client.chat.completions.create.call_count == 1
            assert result == mock_assistant
            # Tool calls should be present but not executed
            assert result.tool_calls is not None
    
    def test_max_turns_one_with_single_tool_call(self, mock_backend_setup, sample_tools):
        """Test max_turns=1 with a single tool call that completes the task."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # First response with tool call
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": 5, "b": 3})
        ])
        
        # Second response after tool execution (final)
        final_response = self.create_mock_final_response("The sum is 8.")
        
        mock_client.chat.completions.create.side_effect = [tool_call_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("I'll add those numbers for you.")
            first_assistant.tool_calls = tool_call_response.choices[0].message.tool_calls
            
            # Mock final assistant response
            final_assistant = Assistant("The sum is 8.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]

            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Add 5 and 3")
            result = llm(messages=messages, tools=sample_tools, max_turns=1)
            
            # Should make two API calls (initial + after tool execution)
            assert mock_client.chat.completions.create.call_count == 2
            assert result == final_assistant
            assert result.tool_calls is None  # Final response has no tool calls

    def test_max_turns_multiple_with_chained_tool_calls(self, mock_backend_setup, sample_tools):
        """Test max_turns=3 with multiple chained tool calls."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # First response: add numbers
        first_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": 5, "b": 3})
        ], "I'll add those numbers first.")
        
        # Second response: get weather
        second_response = self.create_mock_response_with_tool_calls([
            ("get_weather", {"city": "Paris"})
        ], "Now let me check the weather.")
        
        # Final response: no more tool calls
        final_response = self.create_mock_final_response("The sum is 8 and Paris weather is sunny!")
        
        mock_client.chat.completions.create.side_effect = [first_response, second_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant responses
            first_assistant = Assistant("I'll add those numbers first.")
            first_assistant.tool_calls = first_response.choices[0].message.tool_calls
            
            second_assistant = Assistant("Now let me check the weather.")
            second_assistant.tool_calls = second_response.choices[0].message.tool_calls
            
            final_assistant = Assistant("The sum is 8 and Paris weather is sunny!")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, second_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Add 5 and 3, then get weather for Paris")
            result = llm(messages=messages, tools=sample_tools, max_turns=3)
            
            # Should make three API calls
            assert mock_client.chat.completions.create.call_count == 3
            assert result == final_assistant
    
    def test_max_turns_limit_reached(self, mock_backend_setup, sample_tools):
        """Test that execution stops when max_turns limit is reached."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # All responses have tool calls (never ending)
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": 1, "b": 1})
        ], "I'll keep adding numbers.")
        
        mock_client.chat.completions.create.return_value = tool_call_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant response that always has tool calls
            assistant_with_tools = Assistant("I'll keep adding numbers.")
            assistant_with_tools.tool_calls = tool_call_response.choices[0].message.tool_calls
            mock_from_response.return_value = assistant_with_tools
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Keep adding numbers")
            result = llm(messages=messages, tools=sample_tools, max_turns=2)
            
            # Should make exactly max_turns API calls
            assert mock_client.chat.completions.create.call_count == 3
            assert result == assistant_with_tools
            # Should still have tool calls since we hit the limit
            assert result.tool_calls is not None
    
    def test_max_turns_with_text_editor_tool(self, mock_backend_setup):
        """Test max_turns with TEXT_EDITOR_TOOL functionality."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tools with text editor
        tools = Tools([TEXT_EDITOR_TOOL])
        
        # Response with text editor tool call
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("str_replace_based_edit_tool", {
                "command": "str_replace",
                "old_str": "hello",
                "new_str": "hi"
            })
        ], "I'll edit the text for you.")
        
        # Final response
        final_response = self.create_mock_final_response("Text has been updated.")
        
        mock_client.chat.completions.create.side_effect = [tool_call_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant responses
            first_assistant = Assistant("I'll edit the text for you.")
            first_assistant.tool_calls = tool_call_response.choices[0].message.tool_calls
            
            final_assistant = Assistant("Text has been updated.")
            final_assistant.tool_calls = None
            final_assistant.text_files = "hi world"  # Updated text
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Replace 'hello' with 'hi' in the text")
            messages.input_text_files = "hello world"
            
            result = llm(messages=messages, tools=tools, max_turns=1)
            
            # Should make two API calls
            assert mock_client.chat.completions.create.call_count == 2
            assert result == final_assistant
            assert result.text_files == "hi world"
    
    def test_max_turns_invalid_values(self, mock_backend_setup, sample_tools):
        """Test that invalid max_turns values raise appropriate errors."""
        mock_backend_class, mock_client = mock_backend_setup
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Test message")
            
            # Test negative max_turns
            with pytest.raises(ValueError, match="Expected max_turns to be positive integer"):
                llm(messages=messages, tools=sample_tools, max_turns=-1)
    
    def test_tool_execution_error_handling(self, mock_backend_setup, sample_tools):
        """Test error handling during tool execution."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Response with invalid tool call
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("nonexistent_tool", {"param": "value"})
        ])
        
        mock_client.chat.completions.create.return_value = tool_call_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            assistant_with_invalid_tool = Assistant("I'll use a nonexistent tool.")
            assistant_with_invalid_tool.tool_calls = tool_call_response.choices[0].message.tool_calls
            mock_from_response.return_value = assistant_with_invalid_tool
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Use invalid tool")
            
            # Should raise ValueError for unregistered tool
            with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not registered"):
                llm(messages=messages, tools=sample_tools, max_turns=1)
    
    def test_tool_parameter_validation_error(self, mock_backend_setup, sample_tools):
        """Test error handling for invalid tool parameters."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Response with invalid parameters for add_numbers
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": "not_a_number", "b": 3})
        ])
        
        mock_client.chat.completions.create.return_value = tool_call_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            assistant_with_invalid_params = Assistant("I'll add invalid parameters.")
            assistant_with_invalid_params.tool_calls = tool_call_response.choices[0].message.tool_calls
            mock_from_response.return_value = assistant_with_invalid_params
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            messages = Messages() >> User("Add invalid parameters")
            
            # Should raise ValueError for parameter validation error
            with pytest.raises(ValueError, match="Error in tool 'add_numbers' parameters"):
                llm(messages=messages, tools=sample_tools, max_turns=1)
    
    def test_tools_format_compatibility(self, mock_backend_setup):
        """Test that Tools class works with different API formats."""
        mock_backend_class, mock_client = mock_backend_setup
        
        def simple_tool(message: str) -> str:
            """A simple test tool."""
            return f"Processed: {message}"
        
        # Test OpenAI format
        openai_tools = Tools([simple_tool], api_format="openai")
        formatted_tools = openai_tools.format()
        
        assert len(formatted_tools) == 1
        assert formatted_tools[0]["type"] == "function"
        assert formatted_tools[0]["function"]["name"] == "simple_tool"
        
        # Test Anthropic format
        anthropic_tools = Tools([simple_tool], api_format="anthropic")
        formatted_tools = anthropic_tools.format()
        
        assert len(formatted_tools) == 1
        assert "name" in formatted_tools[0]["spec"]
        assert formatted_tools[0]["spec"]["name"] == "simple_tool"
        
        # Test format override
        openai_formatted_as_anthropic = openai_tools.format("anthropic")
        assert len(openai_formatted_as_anthropic) == 1
    
    def test_message_history_management_during_auto_calling(self, mock_backend_setup, sample_tools):
        """Test that message history is properly managed during auto-calling."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # First response with tool call
        tool_call_response = self.create_mock_response_with_tool_calls([
            ("add_numbers", {"a": 10, "b": 20})
        ])
        
        # Final response
        final_response = self.create_mock_final_response("The result is 30.")
        
        mock_client.chat.completions.create.side_effect = [tool_call_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant responses
            first_assistant = Assistant("I'll calculate that for you.")
            first_assistant.tool_calls = tool_call_response.choices[0].message.tool_calls
            
            final_assistant = Assistant("The result is 30.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            initial_messages = Messages() >> System("You are a calculator") >> User("What is 10 + 20?")
            result = llm(messages=initial_messages, tools=sample_tools, max_turns=1)
            
            # Verify the final result
            assert result == final_assistant
            
            # Check that the second API call received the expanded message history
            second_call_args = mock_client.chat.completions.create.call_args_list[1]
            messages_sent = second_call_args[1]['messages']
            
            # Should include: system, user, assistant (with tool calls), tool response
            assert len(messages_sent) >= 4
            
            # Verify message sequence
            roles = [msg['role'] for msg in messages_sent]
            assert 'system' in roles
            assert 'user' in roles
            assert 'assistant' in roles
            assert 'tool' in roles


class TestLLMUnit:
    """Unit tests for LLM class using mocks."""

    @classmethod
    def setup_class(cls):
        os.environ["OPENAI_API_KEY"] = "TEST KEY"
        os.environ["ANTHROPIC_API_KEY"] = "TEST KEY"

    @classmethod
    def teardown_class(cls):
        del os.environ["OPENAI_API_KEY"]
        del os.environ["ANTHROPIC_API_KEY"]

    def test_get_backend_valid_names(self):
        """Test get_backend function with valid provider names."""
        providers = [
            "openai",
            "anthropic",
            "anthropic-bedrock",
            "anthropic-vertex",
            "google",
            "google-genai",
            "together",
            "together-ai",
            "togetherai",
            "fireworks",
            "mistral"
        ]

        for provider in providers:
            backend = get_backend(provider)
            assert backend is not None

    def test_get_backend_invalid_name(self):
        """Test get_backend function with invalid provider name."""
        with pytest.raises(ValueError, match="Unexpected value invalid_provider"):
            get_backend("invalid_provider")

    def test_llm_initialization_defaults(self):
        """Test LLM initialization with default values."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}
        )

        assert llm.backend == "openai"
        assert llm.model == "gpt-4o-mini"
        assert llm.credentials is None
        assert llm.auto_cost_tracking is True
        assert llm.model_params == {}
        assert llm.backend_kwargs == {}
        assert llm.llm_logger is None

    def test_llm_initialization_full(self):
        """Test LLM initialization with all parameters."""
        mock_logger = Mock(spec=LLMLogger)
        model_params = {"temperature": 0.7, "max_tokens": 100}
        backend_kwargs = {"timeout": 30}

        llm = LLM(
            backend="anthropic",
            model="claude-3-haiku-20240307",
            cost={"input": 0.25, "output": 1.25},
            credentials="test-key",
            model_params=model_params,
            backend_kwargs=backend_kwargs,
            logger=mock_logger
        )

        assert llm.backend == "anthropic"
        assert llm.model == "claude-3-haiku-20240307"
        assert llm.credentials == "test-key"
        assert llm.model_params == model_params
        assert llm.backend_kwargs == backend_kwargs
        assert llm.llm_logger == mock_logger

    def test_compute_cost(self):
        """Test cost computation."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}  # per 1M tokens
        )

        # Mock response with usage
        mock_response = Mock()
        mock_response.usage.prompt_tokens = 1000  # 1k tokens
        mock_response.usage.completion_tokens = 500  # 0.5k tokens

        cost = llm.compute_cost(mock_response)

        # Expected: (1000 * 0.15 / 1M) + (500 * 0.60 / 1M) = 0.00015 + 0.0003 = 0.00045
        expected_cost = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
        assert cost == expected_cost

    def test_parse_finish_reason_choices(self):
        """Test parse_finish_reason with choices attribute."""
        mock_output = Mock()
        mock_output.choices = [Mock()]
        mock_output.choices[0].finish_reason = "stop"

        reason = parse_finish_reason(mock_output)
        assert reason == "stop"

    def test_parse_finish_reason_stop_reason(self):
        """Test parse_finish_reason with stop_reason attribute."""
        mock_output = Mock()
        # Remove choices attribute to trigger AttributeError
        del mock_output.choices
        mock_output.stop_reason = "end_turn"

        reason = parse_finish_reason(mock_output)
        assert reason == "end_turn"

    def test_prompt_caching_enabled_true(self):
        """Test prompt caching detection when enabled."""
        llm = LLM(
            backend="anthropic",
            model="claude-3-haiku-20240307",
            cost={"input": 0.25, "output": 1.25},
            model_params={
                "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}
            }
        )

        assert llm.is_prompt_caching_enabled is True

    def test_prompt_caching_enabled_false(self):
        """Test prompt caching detection when disabled."""
        llm = LLM(
            backend="anthropic",
            model="claude-3-haiku-20240307",
            cost={"input": 0.25, "output": 1.25}
        )

        assert llm.is_prompt_caching_enabled is False

    def test_prompt_caching_different_header(self):
        """Test prompt caching detection with different header value."""
        llm = LLM(
            backend="anthropic",
            model="claude-3-haiku-20240307",
            cost={"input": 0.25, "output": 1.25},
            model_params={
                "extra_headers": {"anthropic-beta": "different-value"}
            }
        )

        assert llm.is_prompt_caching_enabled is False

    def test_prepare_messages_prompt_only(self):
        """Test prepare_messages with Prompt object only."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}
        )

        prompt = Prompt(
            user="What is the capital of France?",
            system="You are a geography expert."
        )

        messages = llm.prepare_messages(prompt=prompt)
        message_list = messages.get()

        assert len(message_list) == 2
        assert message_list[0]["role"] == "system"
        assert message_list[0]["content"] == "You are a geography expert."
        assert message_list[1]["role"] == "user"
        assert message_list[1]["content"] == "What is the capital of France?"

    def test_prepare_messages_prompt_no_system(self):
        """Test prepare_messages with Prompt object without system message."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}
        )

        prompt = Prompt(user="Hello!")

        messages = llm.prepare_messages(prompt=prompt)
        message_list = messages.get()

        assert len(message_list) == 1
        assert message_list[0]["role"] == "user"
        assert message_list[0]["content"] == "Hello!"

    def test_prepare_messages_with_messages_object(self):
        """Test prepare_messages with Messages object."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}
        )

        messages = Messages() >> System("Be helpful.") >> User("Hi there!")
        prepared = llm.prepare_messages(messages=messages)
        message_list = prepared.get()

        assert len(message_list) == 2
        assert message_list[0]["role"] == "system"
        assert message_list[0]["content"] == "Be helpful."
        assert message_list[1]["role"] == "user"
        assert message_list[1]["content"] == "Hi there!"

    def test_prepare_messages_claude_prefill(self):
        """Test prepare_messages with Claude model and prefill."""
        llm = LLM(
            backend="anthropic",
            model="claude-3-haiku-20240307",
            cost={"input": 0.25, "output": 1.25}
        )

        messages = Messages() >> User("Complete this: The sky is")
        prepared = llm.prepare_messages(
            messages=messages,
            assistant_prefill="blue"
        )
        message_list = prepared.get()

        # Should have user message + assistant prefill
        assert len(message_list) == 2
        assert message_list[0]["role"] == "user"
        assert message_list[1]["role"] == "assistant"
        assert message_list[1]["content"] == "blue"

    def test_prepare_messages_non_claude_prefill(self):
        """Test prepare_messages with non-Claude model and prefill."""
        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60}
        )

        messages = Messages() >> User("Complete this: The sky is")
        prepared = llm.prepare_messages(
            messages=messages,
            assistant_prefill="blue"
        )
        message_list = prepared.get()

        # Should have user message + assistant prefill + continuation user message
        assert len(message_list) == 3
        assert message_list[0]["role"] == "user"
        assert message_list[1]["role"] == "assistant"
        assert message_list[1]["content"] == "blue"
        assert message_list[2]["role"] == "user"
        assert "continue" in message_list[2]["content"].lower()

    @patch('lmapis.llm.LLM._chat_completion')
    def test_log_interaction_success(self, mock_chat_completion):
        """Test logging of successful interaction."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_chat_completion.return_value = mock_response

        # Setup mock logger
        mock_logger = Mock(spec=LLMLogger)
        mock_logger.is_enabled.return_value = True

        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60},
            logger=mock_logger
        )

        messages = Messages() >> User("Test message")
        result = llm.chat_completion(messages)

        # Verify response is returned
        assert result == mock_response

        # Verify logger was called
        mock_logger.log_interaction.assert_called_once()

        # Verify log entry details
        log_entry = mock_logger.log_interaction.call_args[0][0]
        assert isinstance(log_entry, LogEntry)
        assert log_entry.model == "gpt-4o-mini"
        assert log_entry.backend == "openai"
        assert log_entry.response_content == "Test response"
        assert log_entry.finish_reason == "stop"
        assert log_entry.tokens_prompt == 10
        assert log_entry.tokens_completion == 5
        assert log_entry.error is None

    @patch('lmapis.llm.LLM._chat_completion')
    def test_log_interaction_error(self, mock_chat_completion):
        """Test logging of failed interaction."""
        # Setup mock to raise exception
        test_error = Exception("API Error")
        mock_chat_completion.side_effect = test_error

        # Setup mock logger
        mock_logger = Mock(spec=LLMLogger)
        mock_logger.is_enabled.return_value = True

        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60},
            logger=mock_logger
        )

        messages = Messages() >> User("Test message")

        # Verify exception is raised
        with pytest.raises(Exception, match="API Error"):
            llm.chat_completion(messages)

        # Verify logger was called
        mock_logger.log_interaction.assert_called_once()

        # Verify log entry details
        log_entry = mock_logger.log_interaction.call_args[0][0]
        assert log_entry.error == "API Error"
        assert log_entry.response_content is None

    def test_log_interaction_disabled_logger(self):
        """Test that logging is skipped when logger is disabled."""
        # Setup mock logger that's disabled
        mock_logger = Mock(spec=LLMLogger)
        mock_logger.is_enabled.return_value = False

        llm = LLM(
            backend="openai",
            model="gpt-4o-mini",
            cost={"input": 0.15, "output": 0.60},
            logger=mock_logger
        )

        # Call _log_interaction directly
        llm._log_interaction(
            messages=[],
            parameters={},
            response=None,
            start_time=0,
            end_time=1,
            request_id="test",
            error=None
        )

        # Verify logger was not called
        mock_logger.log_interaction.assert_not_called()

    def test_extract_thinking_content_with_tags(self):
        """Test extraction of thinking content from response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0].message.content = "<think>This is my reasoning</think>This is the answer"

        result_response, reasoning = LLM._extract_thinking_content(mock_response)

        assert reasoning == "This is my reasoning"
        assert result_response.choices[0].message.content == "This is the answer"

    def test_extract_thinking_content_without_tags(self):
        """Test extraction when no thinking tags are present."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is just a regular response"

        result_response, reasoning = LLM._extract_thinking_content(mock_response)

        assert reasoning is None
        assert result_response.choices[
                   0].message.content == "This is just a regular response"

    def test_extract_thinking_content_no_choices(self):
        """Test extraction when response has no choices."""
        mock_response = Mock()
        mock_response.choices = []

        result_response, reasoning = LLM._extract_thinking_content(mock_response)

        assert reasoning is None
        assert result_response == mock_response


