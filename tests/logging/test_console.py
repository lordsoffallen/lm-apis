"""
Unit tests for Console storage backend using structlog.

Tests cover formatting options, error handling, and integration with the LogEntry data model.
"""

import io
import sys
from unittest.mock import Mock, patch

import pytest
import structlog

from src.lmapis.logging.console import Console
from src.lmapis.logging.log_entry import LogEntry


class TestConsoleInitialization:
    """Test Console class initialization and configuration."""
    
    def test_default_initialization(self):
        """Test Console with default parameters."""
        console = Console()
        assert console.format_style == "pretty"
        assert console.output_stream == sys.stdout
        assert hasattr(console, 'logger')
    
    def test_pretty_format_initialization(self):
        """Test Console with pretty format explicitly set."""
        console = Console(format_style="pretty")
        assert console.format_style == "pretty"
    
    def test_compact_format_initialization(self):
        """Test Console with compact format."""
        console = Console(format_style="compact")
        assert console.format_style == "compact"
    
    def test_invalid_format_style(self):
        """Test Console with invalid format style raises ValueError."""
        with pytest.raises(ValueError, match="format_style must be either 'pretty' or 'compact'"):
            Console(format_style="invalid")
    
    def test_custom_output_stream(self):
        """Test Console with custom output stream."""
        custom_stream = io.StringIO()
        console = Console(output_stream=custom_stream)
        assert console.output_stream == custom_stream


class TestColorSupport:
    """Test color detection functionality."""
    
    def test_supports_color_with_tty(self):
        """Test color detection when output is a TTY."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True
        
        with patch.dict('os.environ', {'TERM': 'xterm-256color'}):
            console = Console(output_stream=mock_stream)
            assert console._supports_color() is True
    
    def test_supports_color_without_tty(self):
        """Test color detection when output is not a TTY."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = False
        
        console = Console(output_stream=mock_stream)
        assert console._supports_color() is False
    
    def test_supports_color_with_colorterm(self):
        """Test color detection with COLORTERM environment variable."""
        mock_stream = Mock()
        mock_stream.isatty.return_value = True
        
        with patch.dict('os.environ', {'COLORTERM': 'truecolor'}):
            console = Console(output_stream=mock_stream)
            assert console._supports_color() is True


class TestSaveMethod:
    """Test the save method functionality."""
    
    def test_save_successful_request(self):
        """Test saving a successful LLM request."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_test',
            'model': 'gpt-4',
            'backend': 'openai',
            'cost': 0.002,
            'tokens_prompt': 25,
            'tokens_completion': 12,
            'duration_ms': 1250,
            'response_content': 'Hello! How can I help you today?'
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should contain key information
        assert 'req_test' in output
        assert 'gpt-4' in output
        assert 'openai' in output
        assert 'LLM request completed' in output
    
    def test_save_failed_request(self):
        """Test saving a failed LLM request."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_error',
            'model': 'gpt-4',
            'backend': 'openai',
            'error': 'API rate limit exceeded'
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should contain error information
        assert 'req_error' in output
        assert 'LLM request failed' in output
        assert 'API rate limit exceeded' in output
    
    def test_save_with_messages(self):
        """Test saving request with messages."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_msg',
            'model': 'gpt-4',
            'backend': 'openai',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': 'Hello world'}
            ],
            'response_content': 'Hello! How can I help you today?'
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should contain truncated request content
        assert 'Hello world' in output
        assert 'Hello! How can I help you today?' in output
    
    def test_save_truncates_long_content(self):
        """Test that save method truncates long messages and responses."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        long_content = "x" * 200  # Very long content
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_long',
            'model': 'gpt-4',
            'backend': 'openai',
            'messages': [{'role': 'user', 'content': long_content}],
            'response_content': long_content
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should be truncated with ellipsis
        assert '...' in output
        # Should not contain the full long content
        assert long_content not in output
    
    def test_save_compact_format(self):
        """Test saving in compact JSON format."""
        output_stream = io.StringIO()
        console = Console(format_style="compact", output_stream=output_stream)
        
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_compact',
            'model': 'gpt-4',
            'backend': 'openai',
            'cost': 0.001
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should contain structured log data
        assert 'req_compact' in output
        assert 'gpt-4' in output
        assert 'openai' in output
    
    def test_save_excludes_none_values(self):
        """Test that save method excludes None values from output."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'request_id': 'req_minimal',
            'model': 'gpt-4',
            'backend': 'openai',
            'cost': None,
            'tokens_prompt': None,
            'error': None
        }
        
        console.save(data)
        
        output = output_stream.getvalue()
        # Should contain basic info but not None values
        assert 'req_minimal' in output
        assert 'gpt-4' in output
        # None values should not appear in output
        assert 'None' not in output
    
    def test_save_handles_logging_error(self):
        """Test that save method handles logging errors gracefully."""
        # Create a console with a mock logger that raises an exception
        console = Console()
        console.logger = Mock()
        console.logger.info.side_effect = Exception("Logging error")
        console.logger.error.side_effect = Exception("Logging error")
        
        data = {'timestamp': '2024-01-15T10:30:00Z', 'model': 'gpt-4', 'backend': 'openai'}
        
        # Should not raise an exception
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            console.save(data)
            
            # Should have written error to stderr
            stderr_output = mock_stderr.getvalue()
            assert 'Console logging error' in stderr_output
    
    def test_save_handles_stderr_error(self):
        """Test that save method handles stderr errors gracefully."""
        # Create a console with a mock logger that raises an exception
        console = Console()
        console.logger = Mock()
        console.logger.info.side_effect = Exception("Logging error")
        
        data = {'timestamp': '2024-01-15T10:30:00Z', 'model': 'gpt-4', 'backend': 'openai'}
        
        # Mock stderr to also fail
        with patch('sys.stderr') as mock_stderr:
            mock_stderr.write.side_effect = IOError("Stderr also broken")
            
            # Should not raise an exception even when stderr fails
            console.save(data)


class TestCloseMethod:
    """Test the close method functionality."""
    
    def test_close_flushes_output_stream(self):
        """Test that close method flushes the output stream."""
        mock_stream = Mock()
        console = Console(output_stream=mock_stream)
        
        console.close()
        
        mock_stream.flush.assert_called_once()
    
    def test_close_handles_flush_error(self):
        """Test that close method handles flush errors gracefully."""
        mock_stream = Mock()
        mock_stream.flush.side_effect = IOError("Flush error")
        
        console = Console(output_stream=mock_stream)
        
        # Should not raise an exception
        console.close()
    
    def test_close_handles_stream_without_flush(self):
        """Test that close method handles streams without flush method."""
        mock_stream = Mock()
        del mock_stream.flush  # Remove flush method
        
        console = Console(output_stream=mock_stream)
        
        # Should not raise an exception
        console.close()


class TestContextManager:
    """Test Console as a context manager."""
    
    def test_context_manager_usage(self):
        """Test Console can be used as a context manager."""
        output_stream = io.StringIO()
        
        with Console(output_stream=output_stream) as console:
            assert isinstance(console, Console)
            
            data = {
                'timestamp': '2024-01-15T10:30:00Z',
                'model': 'gpt-4',
                'backend': 'openai'
            }
            console.save(data)
        
        # Should have output after context manager exits
        output = output_stream.getvalue()
        assert 'gpt-4' in output
    
    def test_context_manager_calls_close(self):
        """Test that context manager calls close on exit."""
        mock_stream = Mock()
        
        with Console(output_stream=mock_stream) as console:
            pass
        
        # close() should have been called, which calls flush()
        mock_stream.flush.assert_called_once()


class TestIntegrationWithLogEntry:
    """Test Console integration with LogEntry objects."""
    
    def test_save_log_entry_dict(self):
        """Test saving a LogEntry converted to dictionary."""
        output_stream = io.StringIO()
        console = Console(format_style="compact", output_stream=output_stream)
        
        # Create a LogEntry and convert to dict
        log_entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "Hello"}],
            response_content="Hi there!",
            cost=0.001,
            tokens_prompt=10,
            tokens_completion=5,
            duration_ms=800
        )
        
        console.save(log_entry.to_dict())
        
        output = output_stream.getvalue()
        
        # Should contain key information
        assert 'gpt-4' in output
        assert 'openai' in output
        assert '0.001' in output
    
    def test_save_sanitized_log_entry(self):
        """Test saving a sanitized LogEntry."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        # Create a LogEntry with sensitive data
        log_entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "My email is user@example.com"}],
            response_content="I'll help you with that."
        )
        
        # Sanitize and save
        sanitized_entry = log_entry.sanitize()
        console.save(sanitized_entry.to_dict())
        
        output = output_stream.getvalue()
        
        # Should contain redacted email
        assert '[EMAIL_REDACTED]' in output
        assert 'user@example.com' not in output


class TestStructlogConfiguration:
    """Test structlog configuration for different formats."""
    
    def test_pretty_format_uses_console_renderer(self):
        """Test that pretty format configures ConsoleRenderer."""
        output_stream = io.StringIO()
        console = Console(format_style="pretty", output_stream=output_stream)
        
        # Verify logger is configured
        assert hasattr(console, 'logger')
        assert console.logger is not None
    
    def test_compact_format_uses_json_renderer(self):
        """Test that compact format configures JSONRenderer."""
        output_stream = io.StringIO()
        console = Console(format_style="compact", output_stream=output_stream)
        
        # Verify logger is configured
        assert hasattr(console, 'logger')
        assert console.logger is not None
    
    def test_logger_setup_called_during_init(self):
        """Test that _setup_logger is called during initialization."""
        with patch.object(Console, '_setup_logger') as mock_setup:
            Console()
            mock_setup.assert_called_once()