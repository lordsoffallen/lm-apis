"""
Unit tests for the LLMLogger class.

This module tests the main logging functionality including interaction logging,
retry logging, error handling, and graceful degradation when storage backends fail.
"""

import pytest
import time
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, call

from lmapis.logging import LLMLogger, LoggerConfig, StorageBackend, Console, LogEntry


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.saved_data = []
        self.closed = False
    
    def save(self, data):
        if self.should_fail:
            raise Exception("Mock storage failure")
        self.saved_data.append(data)
    
    def close(self):
        self.closed = True


class TestLLMLoggerInitialization:
    """Test LLMLogger initialization and configuration."""
    
    def test_init_with_valid_config(self):
        """Test initialization with a valid LoggerConfig."""
        config = LoggerConfig()
        logger = LLMLogger(config)
        
        assert logger.config == config
        assert logger.is_enabled() == True
        assert logger._closed == False
        assert len(logger._failed_backends) == 0
    
    def test_init_with_invalid_config(self):
        """Test initialization with invalid config raises ValueError."""
        with pytest.raises(ValueError, match="config must be a LoggerConfig instance"):
            LLMLogger("not a config")
    
    def test_init_with_disabled_config(self):
        """Test initialization with disabled config."""
        config = LoggerConfig.create_disabled()
        logger = LLMLogger(config)
        
        assert logger.is_enabled() == False
    
    def test_is_enabled_when_closed(self):
        """Test is_enabled returns False when logger is closed."""
        config = LoggerConfig()
        logger = LLMLogger(config)
        
        assert logger.is_enabled() == True
        
        logger.close()
        assert logger.is_enabled() == False


class TestLLMLoggerInteractionLogging:
    """Test logging of complete LLM interactions."""
    
    def test_log_interaction_basic(self):
        """Test basic interaction logging with all data included."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'Hello'}],
            parameters={'temperature': 0.7},
            response_content='Hi there!',
            finish_reason='stop',
            cost=0.002,
            tokens_prompt=10,
            tokens_completion=5,
            retry_count=0
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['model'] == 'gpt-4'
        assert logged_data['backend'] == 'openai'
        assert logged_data['messages'] == [{'role': 'user', 'content': 'Hello'}]
        assert logged_data['parameters'] == {'temperature': 0.7}
        assert logged_data['response_content'] == 'Hi there!'
        assert logged_data['finish_reason'] == 'stop'
        assert logged_data['cost'] == 0.002
        assert logged_data['tokens_prompt'] == 10
        assert logged_data['tokens_completion'] == 5
        assert logged_data['retry_count'] == 0
        assert 'request_id' in logged_data
        assert 'timestamp' in logged_data
    
    def test_log_interaction_with_error(self):
        """Test interaction logging when an error occurred."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            error="API rate limit exceeded"
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['error'] == "API rate limit exceeded"
        assert logged_data['model'] == 'gpt-4'
        assert logged_data['backend'] == 'openai'
    
    def test_log_interaction_with_timing(self):
        """Test interaction logging with timing information."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            duration_ms=1500
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['duration_ms'] == 1500
    
    def test_log_interaction_with_custom_request_id(self):
        """Test interaction logging with custom request ID."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        custom_id = "custom_req_123"
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            request_id=custom_id
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['request_id'] == custom_id
    
    def test_log_interaction_with_tool_calls(self):
        """Test interaction logging with tool calls."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}'
                }
            },
            {
                "id": "call_def456",
                "type": "function", 
                "function": {
                    "name": "calculate_sum",
                    "arguments": '{"a": 5, "b": 3}'
                }
            }
        ]
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'What is the weather and what is 5+3?'}],
            response_content='I will help you with both questions.',
            finish_reason='tool_calls',
            tool_calls=tool_calls,
            cost=0.003,
            tokens_prompt=20,
            tokens_completion=8
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['tool_calls'] == tool_calls
        assert logged_data['finish_reason'] == 'tool_calls'
        assert len(logged_data['tool_calls']) == 2
        assert logged_data['tool_calls'][0]['function']['name'] == 'get_weather'
        assert logged_data['tool_calls'][1]['function']['name'] == 'calculate_sum'
        assert logged_data['response_content'] == 'I will help you with both questions.'
    
    def test_log_interaction_disabled_logger(self):
        """Test that disabled logger doesn't log interactions."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig.create_disabled()
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 0
    
    def test_log_interaction_data_filtering(self):
        """Test that data filtering based on config works correctly."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(
            storage_backends=[mock_backend],
            include_request_data=False,
            include_response_data=False,
            include_cost_data=False
        )
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'Hello'}],
            parameters={'temperature': 0.7},
            response_content='Hi there!',
            finish_reason='stop',
            cost=0.002,
            tokens_prompt=10,
            tokens_completion=5
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        # Only model and backend should be included
        assert logged_data['model'] == 'gpt-4'
        assert logged_data['backend'] == 'openai'
        
        # These should be None due to filtering
        assert logged_data.get('messages') is None
        assert logged_data.get('parameters') is None
        assert logged_data.get('response_content') is None
        assert logged_data.get('finish_reason') is None
        assert logged_data.get('tool_calls') is None
        assert logged_data.get('cost') is None
        assert logged_data.get('tokens_prompt') is None
        assert logged_data.get('tokens_completion') is None


class TestLLMLoggerRetryLogging:
    """Test logging of retry attempts."""
    
    def test_log_retry_basic(self):
        """Test basic retry logging."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'Hello'}],
            error="Retry attempt 2: Connection timeout",
            retry_count=2
        )
        
        logger.log_retry(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['model'] == 'gpt-4'
        assert logged_data['backend'] == 'openai'
        assert logged_data['messages'] == [{'role': 'user', 'content': 'Hello'}]
        assert logged_data['error'] == "Retry attempt 2: Connection timeout"
        assert logged_data['retry_count'] == 2
        assert 'request_id' in logged_data
    
    def test_log_retry_with_custom_request_id(self):
        """Test retry logging with custom request ID."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        custom_id = "original_req_123"
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            error="Retry attempt 1: Rate limit",
            retry_count=1,
            request_id=custom_id
        )
        
        logger.log_retry(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['request_id'] == custom_id
        assert logged_data['retry_count'] == 1
    
    def test_log_retry_disabled_logger(self):
        """Test that disabled logger doesn't log retries."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig.create_disabled()
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            error="Retry attempt 1: Error",
            retry_count=1
        )
        
        logger.log_retry(log_entry)
        
        assert len(mock_backend.saved_data) == 0
    
    def test_log_retry_data_filtering(self):
        """Test retry logging respects data filtering configuration."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(
            storage_backends=[mock_backend],
            include_request_data=False
        )
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'Hello'}],
            parameters={'temperature': 0.7},
            error="Retry attempt 1: Error",
            retry_count=1
        )
        
        logger.log_retry(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        assert logged_data['model'] == 'gpt-4'
        assert logged_data['backend'] == 'openai'
        assert logged_data.get('messages') is None
        assert logged_data.get('parameters') is None


class TestLLMLoggerErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_single_backend_failure(self):
        """Test that single backend failure doesn't stop logging to other backends."""
        good_backend = MockStorageBackend()
        bad_backend = MockStorageBackend(should_fail=True)
        
        config = LoggerConfig(storage_backends=[good_backend, bad_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Should not raise exception
        logger.log_interaction(log_entry)
        
        # Good backend should have received the data
        assert len(good_backend.saved_data) == 1
        assert len(bad_backend.saved_data) == 0
        
        # Bad backend should be tracked as failed
        assert bad_backend in logger._failed_backends
        assert good_backend not in logger._failed_backends
    
    def test_all_backends_failure(self):
        """Test behavior when all backends fail."""
        bad_backend1 = MockStorageBackend(should_fail=True)
        bad_backend2 = MockStorageBackend(should_fail=True)
        
        config = LoggerConfig(storage_backends=[bad_backend1, bad_backend2])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Should not raise exception even when all backends fail
        logger.log_interaction(log_entry)
        
        # Both backends should be tracked as failed
        assert bad_backend1 in logger._failed_backends
        assert bad_backend2 in logger._failed_backends
    
    def test_backend_recovery(self):
        """Test that recovered backends are removed from failed tracking."""
        # Start with a failing backend
        backend = MockStorageBackend(should_fail=True)
        config = LoggerConfig(storage_backends=[backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # First call should fail
        logger.log_interaction(log_entry)
        assert backend in logger._failed_backends
        
        # Fix the backend
        backend.should_fail = False
        
        # Second call should succeed and remove from failed tracking
        logger.log_interaction(log_entry)
        assert backend not in logger._failed_backends
        assert len(backend.saved_data) == 1
    
    def test_logging_exception_handling(self):
        """Test that exceptions in logging don't propagate."""
        # Mock a backend that raises an unexpected exception type
        backend = Mock(spec=StorageBackend)
        backend.save.side_effect = RuntimeError("Unexpected error")
        
        config = LoggerConfig(storage_backends=[backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Should not raise exception
        logger.log_interaction(log_entry)
        
        # Backend should be called
        assert backend.save.called
    
    def test_log_entry_filtering_failure(self):
        """Test handling of LogEntry filtering failures."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        # Create a log entry that will cause filtering to fail
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Mock the _apply_data_filtering method to raise an exception
        with patch.object(logger, '_apply_data_filtering', side_effect=Exception("Filtering failed")):
            # Should not raise exception
            logger.log_interaction(log_entry)
            
            # No data should be saved due to filtering failure
            assert len(mock_backend.saved_data) == 0


class TestLLMLoggerBackendManagement:
    """Test storage backend management functionality."""
    
    def test_add_storage_backend(self):
        """Test adding a storage backend."""
        # Start with disabled config to allow empty backends
        config = LoggerConfig.create_disabled()
        config.enabled = True  # Enable after creation to bypass validation
        config.storage_backends = []  # Start with empty list
        logger = LLMLogger(config)
        
        new_backend = MockStorageBackend()
        logger.add_storage_backend(new_backend)
        
        assert new_backend in logger.config.storage_backends
    
    def test_add_invalid_storage_backend(self):
        """Test adding invalid storage backend raises ValueError."""
        config = LoggerConfig()
        logger = LLMLogger(config)
        
        with pytest.raises(ValueError, match="backend must be a StorageBackend instance"):
            logger.add_storage_backend("not a backend")
    
    def test_remove_storage_backend(self):
        """Test removing storage backends by type."""
        backend1 = MockStorageBackend()
        backend2 = Console()
        config = LoggerConfig(storage_backends=[backend1, backend2])
        logger = LLMLogger(config)
        
        # Remove MockStorageBackend type
        removed = logger.remove_storage_backend(MockStorageBackend)
        
        assert removed == True
        assert backend1 not in logger.config.storage_backends
        assert backend2 in logger.config.storage_backends
    
    def test_get_failed_backends(self):
        """Test getting list of failed backends."""
        good_backend = MockStorageBackend()
        bad_backend = MockStorageBackend(should_fail=True)
        
        config = LoggerConfig(storage_backends=[good_backend, bad_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Trigger failure
        logger.log_interaction(log_entry)
        
        failed_backends = logger.get_failed_backends()
        assert bad_backend in failed_backends
        assert good_backend not in failed_backends
    
    def test_reset_failed_backends(self):
        """Test resetting failed backends tracking."""
        bad_backend = MockStorageBackend(should_fail=True)
        config = LoggerConfig(storage_backends=[bad_backend])
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai'
        )
        
        # Trigger failure
        logger.log_interaction(log_entry)
        assert len(logger.get_failed_backends()) == 1
        
        # Reset tracking
        logger.reset_failed_backends()
        assert len(logger.get_failed_backends()) == 0


class TestLLMLoggerContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_usage(self):
        """Test using LLMLogger as a context manager."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        
        with LLMLogger(config) as logger:
            assert logger.is_enabled() == True
            log_entry = LogEntry.create(model='gpt-4', backend='openai')
            logger.log_interaction(log_entry)
        
        # Logger should be closed after context exit
        assert logger._closed == True
        assert mock_backend.closed == True
    
    def test_context_manager_with_exception(self):
        """Test context manager cleanup when exception occurs."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        
        try:
            with LLMLogger(config) as logger:
                log_entry = LogEntry.create(model='gpt-4', backend='openai')
                logger.log_interaction(log_entry)
                raise Exception("Test exception")
        except Exception:
            pass
        
        # Logger should still be closed after exception
        assert logger._closed == True
        assert mock_backend.closed == True


class TestLLMLoggerClose:
    """Test logger cleanup and resource management."""
    
    def test_close_logger(self):
        """Test closing logger and cleaning up resources."""
        mock_backend1 = MockStorageBackend()
        mock_backend2 = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend1, mock_backend2])
        logger = LLMLogger(config)
        
        logger.close()
        
        assert logger._closed == True
        assert mock_backend1.closed == True
        assert mock_backend2.closed == True
        assert len(logger._failed_backends) == 0
    
    def test_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        # Close multiple times
        logger.close()
        logger.close()
        logger.close()
        
        assert logger._closed == True
        assert mock_backend.closed == True
    
    def test_close_with_backend_errors(self):
        """Test close handling when backends raise exceptions."""
        # Mock backend that fails on close
        mock_backend = Mock(spec=StorageBackend)
        mock_backend.close.side_effect = Exception("Close failed")
        
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        # Should not raise exception
        logger.close()
        
        assert logger._closed == True
        assert mock_backend.close.called


class TestLLMLoggerSanitization:
    """Test data sanitization functionality."""
    
    def test_sanitization_enabled(self):
        """Test that sanitization is applied when enabled."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend], sanitize_messages=True)
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'My email is test@example.com'}]
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        # Email should be sanitized
        assert 'test@example.com' not in str(logged_data['messages'])
        assert '[EMAIL_REDACTED]' in str(logged_data['messages'])
    
    def test_sanitization_disabled(self):
        """Test that sanitization is not applied when disabled."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend], sanitize_messages=False)
        logger = LLMLogger(config)
        
        log_entry = LogEntry.create(
            model='gpt-4',
            backend='openai',
            messages=[{'role': 'user', 'content': 'My email is test@example.com'}]
        )
        
        logger.log_interaction(log_entry)
        
        assert len(mock_backend.saved_data) == 1
        logged_data = mock_backend.saved_data[0]
        
        # Email should not be sanitized
        assert 'test@example.com' in str(logged_data['messages'])


class TestLLMLoggerRepr:
    """Test string representation of LLMLogger."""
    
    def test_repr(self):
        """Test __repr__ method."""
        mock_backend = MockStorageBackend()
        config = LoggerConfig(storage_backends=[mock_backend])
        logger = LLMLogger(config)
        
        repr_str = repr(logger)
        
        assert 'LLMLogger' in repr_str
        assert 'enabled=True' in repr_str
        assert 'MockStorageBackend' in repr_str
        assert 'failed_backends=0' in repr_str
        assert 'closed=False' in repr_str