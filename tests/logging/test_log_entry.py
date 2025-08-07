"""
Unit tests for the LogEntry dataclass and related functionality.

Tests cover LogEntry creation, sanitization, JSON serialization,
and all the requirements specified in the task.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import patch

from lmapis.logging.log_entry import LogEntry


class TestLogEntryCreation:
    """Test LogEntry creation and basic functionality."""
    
    def test_create_minimal_log_entry(self):
        """Test creating a LogEntry with minimal required fields."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai"
        )
        
        assert entry.model == "gpt-4"
        assert entry.backend == "openai"
        assert isinstance(entry.timestamp, datetime)
        assert entry.request_id.startswith("req_")
        assert len(entry.request_id) == 12  # "req_" + 8 hex chars
    
    def test_create_complete_log_entry(self):
        """Test creating a LogEntry with all fields populated."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello world"}
        ]
        parameters = {"temperature": 0.7, "max_tokens": 150}
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=messages,
            parameters=parameters,
            response_content="Hello! How can I help you today?",
            finish_reason="stop",
            cost=0.002,
            tokens_prompt=25,
            tokens_completion=12,
            duration_ms=1250,
            retry_count=0
        )
        
        assert entry.model == "gpt-4"
        assert entry.backend == "openai"
        assert entry.messages == messages
        assert entry.parameters == parameters
        assert entry.response_content == "Hello! How can I help you today?"
        assert entry.finish_reason == "stop"
        assert entry.cost == 0.002
        assert entry.tokens_prompt == 25
        assert entry.tokens_completion == 12
        assert entry.duration_ms == 1250
        assert entry.retry_count == 0
        assert entry.error is None
    
    def test_create_error_log_entry(self):
        """Test creating a LogEntry for a failed request."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            error="Rate limit exceeded",
            retry_count=3
        )
        
        assert entry.model == "gpt-4"
        assert entry.backend == "openai"
        assert entry.error == "Rate limit exceeded"
        assert entry.retry_count == 3
        assert entry.response_content is None
    
    def test_custom_timestamp_and_request_id(self):
        """Test creating a LogEntry with custom timestamp and request ID."""
        custom_timestamp = datetime(2024, 1, 15, 10, 30, 0)
        custom_request_id = "custom_req_123"
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            timestamp=custom_timestamp,
            request_id=custom_request_id
        )
        
        assert entry.timestamp == custom_timestamp
        assert entry.request_id == custom_request_id


class TestLogEntrySanitization:
    """Test data sanitization functionality."""
    
    def test_sanitize_api_keys(self):
        """Test sanitization of API keys in various formats."""
        messages = [
            {"role": "user", "content": "My API key is sk-1234567890abcdef1234567890abcdef1234567890abcdef"},
            {"role": "user", "content": "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz1234567890abcdef"}
        ]
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=messages,
            response_content="Here's your API key: sk-9876543210fedcba9876543210fedcba9876543210fedcba"
        )
        
        sanitized = entry.sanitize()
        
        # Check that API keys are redacted in messages
        assert "[API_KEY_REDACTED]" in sanitized.messages[0]["content"]
        assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in sanitized.messages[0]["content"]
        
        # Check that API keys are redacted in response
        assert "[API_KEY_REDACTED]" in sanitized.response_content
        assert "sk-9876543210fedcba9876543210fedcba9876543210fedcba" not in sanitized.response_content
    
    def test_sanitize_email_addresses(self):
        """Test sanitization of email addresses."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "Contact me at john.doe@example.com"}],
            response_content="I'll send it to jane.smith@company.org"
        )
        
        sanitized = entry.sanitize()
        
        assert "[EMAIL_REDACTED]" in sanitized.messages[0]["content"]
        assert "john.doe@example.com" not in sanitized.messages[0]["content"]
        assert "[EMAIL_REDACTED]" in sanitized.response_content
        assert "jane.smith@company.org" not in sanitized.response_content
    
    def test_sanitize_phone_numbers(self):
        """Test sanitization of phone numbers in various formats."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[
                {"role": "user", "content": "Call me at 555-123-4567 or (555) 987-6543"},
                {"role": "user", "content": "My number is 5551234567"}
            ]
        )
        
        sanitized = entry.sanitize()
        
        assert "[PHONE_REDACTED]" in sanitized.messages[0]["content"]
        assert "555-123-4567" not in sanitized.messages[0]["content"]
        assert "(555) 987-6543" not in sanitized.messages[0]["content"]
        assert "5551234567" not in sanitized.messages[1]["content"]
    
    def test_sanitize_credit_cards(self):
        """Test sanitization of credit card numbers."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "My card is 4532-1234-5678-9012"}],
            response_content="Card ending in 1234 5678 9012 3456"
        )
        
        sanitized = entry.sanitize()
        
        assert "[CARD_REDACTED]" in sanitized.messages[0]["content"]
        assert "4532-1234-5678-9012" not in sanitized.messages[0]["content"]
        assert "[CARD_REDACTED]" in sanitized.response_content
        assert "1234 5678 9012 3456" not in sanitized.response_content
    
    def test_sanitize_ssn(self):
        """Test sanitization of Social Security Numbers."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
        )
        
        sanitized = entry.sanitize()
        
        assert "[SSN_REDACTED]" in sanitized.messages[0]["content"]
        assert "123-45-6789" not in sanitized.messages[0]["content"]
    
    def test_sanitize_nested_data(self):
        """Test sanitization of nested data structures."""
        parameters = {
            "temperature": 0.7,
            "metadata": {
                "user_email": "test@example.com",
                "api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef"
            }
        }
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            parameters=parameters
        )
        
        sanitized = entry.sanitize()
        
        assert sanitized.parameters["temperature"] == 0.7  # Non-sensitive data preserved
        assert "[EMAIL_REDACTED]" in str(sanitized.parameters["metadata"])
        assert "[API_KEY_REDACTED]" in str(sanitized.parameters["metadata"])
        assert "test@example.com" not in str(sanitized.parameters)
    
    def test_custom_sanitizer(self):
        """Test custom sanitization function."""
        def custom_sanitizer(data):
            # Replace all instances of "secret" with "[CUSTOM_REDACTED]"
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        data[key] = value.replace("secret", "[CUSTOM_REDACTED]")
            return data
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            response_content="The secret code is secret123"
        )
        
        sanitized = entry.sanitize(
            remove_sensitive_patterns=False,
            custom_sanitizer=custom_sanitizer
        )
        
        assert "[CUSTOM_REDACTED]" in sanitized.response_content
        assert "secret" not in sanitized.response_content
    
    def test_sanitize_preserves_original(self):
        """Test that sanitization creates a new instance and preserves the original."""
        original_content = "Contact me at john@example.com"
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            response_content=original_content
        )
        
        sanitized = entry.sanitize()
        
        # Original should be unchanged
        assert entry.response_content == original_content
        # Sanitized should be different
        assert sanitized.response_content != original_content
        assert "[EMAIL_REDACTED]" in sanitized.response_content


class TestLogEntryJSONSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            timestamp=timestamp,
            request_id="test_req_123"
        )
        
        data = entry.to_dict()
        
        assert data["model"] == "gpt-4"
        assert data["backend"] == "openai"
        assert data["timestamp"] == "2024-01-15T10:30:00Z"
        assert data["request_id"] == "test_req_123"
        
        # None values should be excluded
        assert "messages" not in data
        assert "error" not in data
    
    def test_to_dict_complete(self):
        """Test dictionary conversion with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        parameters = {"temperature": 0.7}
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=messages,
            parameters=parameters,
            response_content="Hi there!",
            finish_reason="stop",
            cost=0.001,
            tokens_prompt=10,
            tokens_completion=5,
            duration_ms=500,
            retry_count=0
        )
        
        data = entry.to_dict()
        
        assert data["messages"] == messages
        assert data["parameters"] == parameters
        assert data["response_content"] == "Hi there!"
        assert data["finish_reason"] == "stop"
        assert data["cost"] == 0.001
        assert data["tokens_prompt"] == 10
        assert data["tokens_completion"] == 5
        assert data["duration_ms"] == 500
        assert data["retry_count"] == 0
    
    def test_to_json_compact(self):
        """Test compact JSON serialization."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            response_content="Hello world"
        )
        
        json_str = entry.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model"] == "gpt-4"
        assert parsed["backend"] == "openai"
        assert parsed["response_content"] == "Hello world"
        
        # Should be compact (no indentation)
        assert "\n" not in json_str
    
    def test_to_json_indented(self):
        """Test indented JSON serialization."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            response_content="Hello world"
        )
        
        json_str = entry.to_json(indent=2)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model"] == "gpt-4"
        
        # Should be indented
        assert "\n" in json_str
        assert "  " in json_str  # 2-space indentation
    
    def test_from_dict(self):
        """Test creating LogEntry from dictionary."""
        data = {
            "timestamp": "2024-01-15T10:30:00Z",
            "request_id": "test_req_123",
            "model": "gpt-4",
            "backend": "openai",
            "response_content": "Hello world",
            "cost": 0.001
        }
        
        entry = LogEntry.from_dict(data)
        
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.year == 2024
        assert entry.timestamp.month == 1
        assert entry.timestamp.day == 15
        assert entry.request_id == "test_req_123"
        assert entry.model == "gpt-4"
        assert entry.backend == "openai"
        assert entry.response_content == "Hello world"
        assert entry.cost == 0.001
    
    def test_from_json(self):
        """Test creating LogEntry from JSON string."""
        json_str = '''
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "request_id": "test_req_123",
            "model": "gpt-4",
            "backend": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_content": "Hi there!",
            "cost": 0.001
        }
        '''
        
        entry = LogEntry.from_json(json_str)
        
        assert isinstance(entry.timestamp, datetime)
        assert entry.request_id == "test_req_123"
        assert entry.model == "gpt-4"
        assert entry.backend == "openai"
        assert entry.messages == [{"role": "user", "content": "Hello"}]
        assert entry.response_content == "Hi there!"
        assert entry.cost == 0.001
    
    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization preserve data."""
        original_entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": "Test message"}],
            parameters={"temperature": 0.7, "max_tokens": 100},
            response_content="Test response",
            finish_reason="stop",
            cost=0.002,
            tokens_prompt=15,
            tokens_completion=8,
            duration_ms=750,
            retry_count=1
        )
        
        # Convert to JSON and back
        json_str = original_entry.to_json()
        restored_entry = LogEntry.from_json(json_str)
        
        # Compare all fields
        assert restored_entry.model == original_entry.model
        assert restored_entry.backend == original_entry.backend
        assert restored_entry.messages == original_entry.messages
        assert restored_entry.parameters == original_entry.parameters
        assert restored_entry.response_content == original_entry.response_content
        assert restored_entry.finish_reason == original_entry.finish_reason
        assert restored_entry.cost == original_entry.cost
        assert restored_entry.tokens_prompt == original_entry.tokens_prompt
        assert restored_entry.tokens_completion == original_entry.tokens_completion
        assert restored_entry.duration_ms == original_entry.duration_ms
        assert restored_entry.retry_count == original_entry.retry_count
        
        # Timestamps should be equal (within microsecond precision)
        assert abs((restored_entry.timestamp - original_entry.timestamp).total_seconds()) < 0.001


class TestLogEntryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_messages_list(self):
        """Test handling of empty messages list."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[]
        )
        
        assert entry.messages == []
        
        # Should serialize and deserialize correctly
        json_str = entry.to_json()
        restored = LogEntry.from_json(json_str)
        assert restored.messages == []
    
    def test_none_values_excluded_from_dict(self):
        """Test that None values are excluded from dictionary representation."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai"
        )
        
        data = entry.to_dict()
        
        # Only non-None values should be present
        expected_keys = {"timestamp", "request_id", "model", "backend"}
        assert set(data.keys()) == expected_keys
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content in messages and responses."""
        unicode_content = "Hello 世界! 🌍 Café naïve résumé"
        
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            messages=[{"role": "user", "content": unicode_content}],
            response_content=unicode_content
        )
        
        # Should serialize and deserialize correctly
        json_str = entry.to_json()
        restored = LogEntry.from_json(json_str)
        
        assert restored.messages[0]["content"] == unicode_content
        assert restored.response_content == unicode_content
    
    def test_large_numbers_handling(self):
        """Test handling of large numbers in cost and token fields."""
        entry = LogEntry.create(
            model="gpt-4",
            backend="openai",
            cost=999999.999999,
            tokens_prompt=1000000,
            tokens_completion=2000000,
            duration_ms=9999999
        )
        
        # Should serialize and deserialize correctly
        json_str = entry.to_json()
        restored = LogEntry.from_json(json_str)
        
        assert restored.cost == 999999.999999
        assert restored.tokens_prompt == 1000000
        assert restored.tokens_completion == 2000000
        assert restored.duration_ms == 9999999