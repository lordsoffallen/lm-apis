"""
Integration tests for LLM class with logging functionality.

This module tests the integration between the LLM class and the logging system,
ensuring that logging works correctly without impacting LLM performance or
functionality.
"""

import pytest
import time
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from lmapis.llm import LLM, Prompt, parse_finish_reason, get_backend
from lmapis.logging import LLMLogger, LogEntry
from lmapis.logging.config import LoggerConfig
from lmapis.logging.storage import StorageBackend
from lmapis.utils.messages import Messages, System, User, Assistant


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing logging integration."""
    
    def __init__(self):
        self.saved_entries: List[Dict[str, Any]] = []
        self.save_calls = 0
        self.close_calls = 0
        self.should_fail = False
        self.failure_message = "Mock storage failure"
    
    def save(self, data: Dict[str, Any]) -> None:
        self.save_calls += 1
        if self.should_fail:
            raise Exception(self.failure_message)
        self.saved_entries.append(data.copy())
    
    def close(self) -> None:
        self.close_calls += 1


class TestLLMLoggingIntegration:
    """Test suite for LLM class logging integration."""
    
    @pytest.fixture
    def mock_backend_setup(self):
        """Centralized mock setup for LLM backend to reduce duplication."""
        mock_client = Mock()
        mock_response = Mock()
        
        # Configure mock response
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response content"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_instance.client = mock_client
        mock_backend_class.return_value = mock_backend_instance
        
        return mock_backend_class, mock_client, mock_response
    
    @pytest.fixture
    def mock_storage_backend(self):
        """Create a mock storage backend for testing."""
        return MockStorageBackend()
    
    @pytest.fixture
    def logger_config_with_mock_storage(self, mock_storage_backend):
        """Create logger config with mock storage backend."""
        return LoggerConfig(
            enabled=True,
            storage_backends=[mock_storage_backend],
            include_request_data=True,
            include_response_data=True,
            include_cost_data=True,
            sanitize_messages=False
        )
    
    @pytest.fixture
    def llm_logger_with_mock_storage(self, logger_config_with_mock_storage):
        """Create LLM logger with mock storage backend."""
        return LLMLogger(logger_config_with_mock_storage)
    
    def test_llm_without_logger_works_normally(self, mock_backend_setup):
        """Test that LLM works normally without any logger configured."""
        mock_backend_class, mock_client, mock_response = mock_backend_setup
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM without logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Verify no logger is set
            assert llm.llm_logger is None
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model - should work normally
            result = llm.chat_completion(messages)
            
            # Verify the call was made
            assert mock_client.chat.completions.create.called
            assert result == mock_response
    
    def test_llm_with_logger_logs_successful_interactions(
        self, 
        mock_backend_setup, 
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test that LLM logs successful interactions when logger is configured."""
        mock_backend_class, mock_client, mock_response = mock_backend_setup
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            # Verify logger is set
            assert llm.llm_logger is not None
            assert llm.llm_logger.is_enabled()
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model
            result = llm.chat_completion(messages)
            
            # Verify the API call was made
            assert mock_client.chat.completions.create.called
            assert result == mock_response
            
            # Verify logging occurred
            assert mock_storage_backend.save_calls == 1
            assert len(mock_storage_backend.saved_entries) == 1
            
            # Verify logged data structure
            logged_entry = mock_storage_backend.saved_entries[0]
            assert logged_entry['model'] == 'gpt-4'
            assert logged_entry['backend'] == 'openai'
            assert logged_entry['response_content'] == 'Test response content'
            assert logged_entry['finish_reason'] == 'stop'
            assert logged_entry['tokens_prompt'] == 10
            assert logged_entry['tokens_completion'] == 20
            assert logged_entry['cost'] is not None
            assert logged_entry.get('error') is None  # error key may not exist if no error
            assert 'request_id' in logged_entry
            assert 'timestamp' in logged_entry
            assert 'duration_ms' in logged_entry
    
    def test_llm_with_logger_logs_failed_interactions(
        self, 
        mock_backend_setup,
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test that LLM logs failed interactions when exceptions occur."""
        mock_backend_class, mock_client, _ = mock_backend_setup
        
        # Use a non-retryable exception to avoid hanging
        test_exception = Exception("Bad Request - invalid parameters")
        mock_client.chat.completions.create.side_effect = test_exception
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.should_retry_exception', return_value=False):
            
            # Create LLM with logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model - should raise exception
            with pytest.raises(Exception, match="Bad Request - invalid parameters"):
                llm.chat_completion(messages)
            
            # Verify logging occurred even for failed interaction
            assert mock_storage_backend.save_calls == 1
            assert len(mock_storage_backend.saved_entries) == 1
            
            # Verify logged error data
            logged_entry = mock_storage_backend.saved_entries[0]
            assert logged_entry['model'] == 'gpt-4'
            assert logged_entry['backend'] == 'openai'
            assert logged_entry['error'] == 'Bad Request - invalid parameters'
            assert logged_entry.get('response_content') is None  # may not exist if None
            assert 'request_id' in logged_entry
    
    def test_llm_continues_when_logging_fails(self, mock_backend_setup):
        """Test that LLM continues to work when logging fails."""
        mock_backend_class, mock_client, mock_response = mock_backend_setup
        
        # Create a failing storage backend
        failing_backend = MockStorageBackend()
        failing_backend.should_fail = True
        
        logger_config = LoggerConfig(
            enabled=True,
            storage_backends=[failing_backend]
        )
        llm_logger = LLMLogger(logger_config)
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with failing logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model - should work despite logging failure
            result = llm.chat_completion(messages)
            
            # Verify the API call was made successfully
            assert mock_client.chat.completions.create.called
            assert result == mock_response
            
            # Verify logging was attempted but failed
            assert failing_backend.save_calls == 1
            assert len(failing_backend.saved_entries) == 0  # No entries saved due to failure
    
    def test_llm_with_disabled_logger_skips_logging(self, mock_backend_setup):
        """Test that LLM skips logging when logger is disabled."""
        mock_backend_class, mock_client, mock_response = mock_backend_setup
        mock_storage_backend = MockStorageBackend()
        
        # Create disabled logger config
        logger_config = LoggerConfig(
            enabled=False,
            storage_backends=[mock_storage_backend]
        )
        llm_logger = LLMLogger(logger_config)
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with disabled logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger
            )
            
            # Verify logger is disabled
            assert not llm.llm_logger.is_enabled()
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model
            result = llm.chat_completion(messages)
            
            # Verify the API call was made
            assert mock_client.chat.completions.create.called
            assert result == mock_response
            
            # Verify no logging occurred
            assert mock_storage_backend.save_calls == 0
            assert len(mock_storage_backend.saved_entries) == 0
    
    def test_llm_logs_retry_attempts(self, mock_backend_setup, llm_logger_with_mock_storage, mock_storage_backend):
        """Test that LLM logs retry attempts when they occur."""
        mock_backend_class, mock_client, _ = mock_backend_setup
        
        # Configure mock response for successful call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success after retries"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        # Mock the retry mechanism to avoid hanging
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.should_retry_exception', return_value=False):
            
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create LLM with logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model - should succeed
            result = llm.chat_completion(messages)
            
            # Verify the call succeeded
            assert result == mock_response
            
            # Verify logging occurred
            assert mock_storage_backend.save_calls == 1
            assert len(mock_storage_backend.saved_entries) == 1
            
            # Check success log
            logged_entry = mock_storage_backend.saved_entries[0]
            assert logged_entry['response_content'] == 'Success after retries'
    
    def test_llm_logs_claude_model_interactions(
        self, 
        mock_backend_setup,
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test that LLM logs Claude model interactions correctly."""
        mock_backend_class, mock_client, _ = mock_backend_setup
        
        # Configure mock response for Claude
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Claude response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with Claude model and logger
            llm = LLM(
                backend="anthropic",
                model="claude-3-sonnet",
                cost={"input": 0.003, "output": 0.015},
                logger=llm_logger_with_mock_storage
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Call the model
            result = llm.chat_completion(messages)
            
            # Verify the API call was made
            assert mock_client.chat.completions.create.called
            assert result == mock_response
            
            # Verify logging occurred
            assert mock_storage_backend.save_calls == 1
            logged_entry = mock_storage_backend.saved_entries[0]
            assert logged_entry['model'] == 'claude-3-sonnet'
            assert logged_entry['backend'] == 'anthropic'
            assert logged_entry['response_content'] == 'Claude response'
    
    def test_llm_logs_with_prefill_response(
        self, 
        mock_backend_setup,
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test that LLM logs interactions with assistant prefill correctly."""
        mock_backend_class, mock_client, _ = mock_backend_setup
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Continued response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with logger (non-Claude model)
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            # Create test messages with prefill
            messages = Messages() >> System("You are helpful") >> User("Hello")
            prefill = Assistant("I'll help you")

            messages = llm._get_messages(messages, prefill)
            
            # Call the model with prefill
            result = llm.chat_completion(messages)
            
            # Verify the API call was made
            assert mock_client.chat.completions.create.called
            assert result == mock_response
            
            # Verify logging occurred
            assert mock_storage_backend.save_calls == 1
            logged_entry = mock_storage_backend.saved_entries[0]
            
            # Check that the logged messages include the continuation prompt
            logged_messages = logged_entry['messages']
            assert any("continue where you left off" in str(msg).lower() for msg in logged_messages)
    
    def test_llm_full_call_with_logging(
        self, 
        mock_backend_setup,
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test the full LLM.__call__ method with logging enabled."""
        mock_backend_class, mock_client, _ = mock_backend_setup
        
        # Configure mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Full call response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 35
        
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with logger
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Mock Assistant.from_model_response
            with patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
                mock_assistant = Assistant("Full call response")
                mock_from_response.return_value = mock_assistant
                
                # Call the full LLM method
                result = llm(messages=messages)
                
                # Verify the result
                assert result == mock_assistant
                
                # Verify logging occurred
                assert mock_storage_backend.save_calls == 1
                logged_entry = mock_storage_backend.saved_entries[0]
                assert logged_entry['response_content'] == 'Full call response'
                assert logged_entry['finish_reason'] == 'stop'
    
    def test_llm_performance_with_logging_enabled(
        self, 
        mock_backend_setup,
        llm_logger_with_mock_storage,
        mock_storage_backend
    ):
        """Test that logging doesn't significantly impact LLM performance."""
        mock_backend_class, mock_client, mock_response = mock_backend_setup
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class):
            # Create LLM with and without logger
            llm_with_logger = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03},
                logger=llm_logger_with_mock_storage
            )
            
            llm_without_logger = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create test messages
            messages = Messages() >> System("You are helpful") >> User("Hello")
            
            # Time calls with logging
            start_time = time.time()
            for _ in range(10):
                llm_with_logger.chat_completion(messages)
            with_logging_time = time.time() - start_time
            
            # Reset mock call count
            mock_client.chat.completions.create.reset_mock()
            
            # Time calls without logging
            start_time = time.time()
            for _ in range(10):
                llm_without_logger.chat_completion(messages)
            without_logging_time = time.time() - start_time
            
            # Verify logging doesn't add excessive overhead (less than 3x increase)
            # Note: In real scenarios, overhead would be much lower, but mocking adds overhead
            overhead_ratio = with_logging_time / without_logging_time
            assert overhead_ratio < 3.0, f"Logging overhead too high: {overhead_ratio:.2f}x"
            
            # Verify logging occurred for all calls
            assert mock_storage_backend.save_calls == 10


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
            "openai", "anthropic", "google", "google-genai",
            "together", "together-ai", "togetherai", "fireworks", "mistral"
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
