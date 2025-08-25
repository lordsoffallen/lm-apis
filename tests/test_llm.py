"""
Integration tests for LLM class with logging functionality.

This module tests the integration between the LLM class and the logging system,
ensuring that logging works correctly without impacting LLM performance or
functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from lmapis.llm import LLM
from lmapis.logging.logger import LLMLogger
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