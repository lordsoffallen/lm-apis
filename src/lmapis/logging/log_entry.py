"""
Core data structures for flexible LLM logging system.

This module contains the LogEntry dataclass and related utilities for capturing
complete LLM interaction data including requests, responses, costs, and errors.
"""

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class LogEntry:
    """
    Data model for complete LLM request/response/cost information.
    
    This class captures all relevant information from an LLM interaction
    including request parameters, response data, cost tracking, and error
    information in a single structured log entry.
    """
    
    # Core identification
    timestamp: datetime
    request_id: str
    model: str
    backend: str
    
    # Request data
    messages: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    # Response data
    response_content: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Cost and performance metrics
    cost: Optional[float] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    duration_ms: Optional[int] = None
    
    # Error information
    error: Optional[str] = None
    retry_count: Optional[int] = None
    
    @classmethod
    def create(
        cls,
        model: str,
        backend: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        response_content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        cost: Optional[float] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        retry_count: Optional[int] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "LogEntry":
        """
        Create a new LogEntry with automatic timestamp and request ID generation.
        
        Args:
            model: The LLM model name
            backend: The backend provider (e.g., 'openai', 'anthropic')
            messages: List of message dictionaries from the request
            parameters: Request parameters (temperature, max_tokens, etc.)
            response_content: The generated response content
            finish_reason: Why the model stopped generating
            cost: Total cost of the request in dollars
            tokens_prompt: Number of prompt tokens used
            tokens_completion: Number of completion tokens generated
            duration_ms: Request duration in milliseconds
            error: Error message if the request failed
            retry_count: Number of retry attempts made
            request_id: Unique identifier for the request (auto-generated if None)
            timestamp: Request timestamp (auto-generated if None)
            
        Returns:
            A new LogEntry instance
        """
        return cls(
            timestamp=timestamp or datetime.utcnow(),
            request_id=request_id or f"req_{uuid.uuid4().hex[:8]}",
            model=model,
            backend=backend,
            messages=messages,
            parameters=parameters,
            response_content=response_content,
            finish_reason=finish_reason,
            cost=cost,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            duration_ms=duration_ms,
            error=error,
            retry_count=retry_count,
        )
    
    def sanitize(
        self, 
        remove_sensitive_patterns: bool = True,
        custom_sanitizer: Optional[callable] = None
    ) -> "LogEntry":
        """
        Create a sanitized copy of the log entry with sensitive information removed.
        
        Args:
            remove_sensitive_patterns: Whether to apply default sensitive data patterns
            custom_sanitizer: Optional custom function to sanitize data
            
        Returns:
            A new LogEntry instance with sanitized data
        """
        # Create a deep copy of the current entry
        sanitized_data = self.to_dict()
        
        if remove_sensitive_patterns:
            sanitized_data = self._apply_default_sanitization(sanitized_data)
        
        if custom_sanitizer:
            sanitized_data = custom_sanitizer(sanitized_data)
        
        # Create new LogEntry from sanitized data
        return LogEntry(
            timestamp=sanitized_data["timestamp"],
            request_id=sanitized_data["request_id"],
            model=sanitized_data["model"],
            backend=sanitized_data["backend"],
            messages=sanitized_data.get("messages"),
            parameters=sanitized_data.get("parameters"),
            response_content=sanitized_data.get("response_content"),
            finish_reason=sanitized_data.get("finish_reason"),
            cost=sanitized_data.get("cost"),
            tokens_prompt=sanitized_data.get("tokens_prompt"),
            tokens_completion=sanitized_data.get("tokens_completion"),
            duration_ms=sanitized_data.get("duration_ms"),
            error=sanitized_data.get("error"),
            retry_count=sanitized_data.get("retry_count"),
        )
    
    def _apply_default_sanitization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default sanitization patterns to remove sensitive information.
        
        This method removes common sensitive patterns like:
        - API keys
        - Email addresses
        - Phone numbers
        - Credit card numbers
        - Social security numbers
        - Personal names in certain contexts
        
        Args:
            data: Dictionary representation of the log entry
            
        Returns:
            Sanitized dictionary with sensitive data replaced
        """
        # Patterns for sensitive data
        sensitive_patterns = [
            # API keys and tokens
            (r'sk-[a-zA-Z0-9]{48}', '[API_KEY_REDACTED]'),  # OpenAI API keys
            (r'Bearer [a-zA-Z0-9\-_\.]+', 'Bearer [TOKEN_REDACTED]'),  # Bearer tokens
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]+', 'api_key: [REDACTED]'),
            
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
            
            # Phone numbers (US format)
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]'),
            (r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '[PHONE_REDACTED]'),
            
            # Credit card numbers (basic pattern)
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REDACTED]'),
            
            # Social Security Numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
        ]
        
        # Apply sanitization to string fields
        for field_name in ['messages', 'parameters', 'response_content', 'error']:
            if field_name in data and data[field_name] is not None:
                data[field_name] = self._sanitize_field(data[field_name], sensitive_patterns)
        
        return data
    
    def _sanitize_field(self, field_value: Any, patterns: List[tuple]) -> Any:
        """
        Recursively sanitize a field value using the provided patterns.
        
        Args:
            field_value: The value to sanitize
            patterns: List of (pattern, replacement) tuples
            
        Returns:
            Sanitized field value
        """
        if isinstance(field_value, str):
            # Apply all patterns to string values
            for pattern, replacement in patterns:
                field_value = re.sub(pattern, replacement, field_value, flags=re.IGNORECASE)
            return field_value
        
        elif isinstance(field_value, dict):
            # Recursively sanitize dictionary values
            return {
                key: self._sanitize_field(value, patterns)
                for key, value in field_value.items()
            }
        
        elif isinstance(field_value, list):
            # Recursively sanitize list items
            return [
                self._sanitize_field(item, patterns)
                for item in field_value
            ]
        
        else:
            # Return non-string values as-is
            return field_value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the LogEntry to a dictionary with proper datetime serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        data = asdict(self)
        
        # Convert datetime to ISO format string for JSON serialization
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat() + 'Z'
        
        # Remove None values to keep the output clean
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert the LogEntry to a JSON string with proper datetime handling.
        
        Args:
            indent: Number of spaces for JSON indentation (None for compact)
            
        Returns:
            JSON string representation of the log entry
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """
        Create a LogEntry from a dictionary representation.
        
        Args:
            data: Dictionary containing log entry data
            
        Returns:
            LogEntry instance
        """
        # Parse timestamp if it's a string
        if isinstance(data.get('timestamp'), str):
            timestamp_str = data['timestamp'].rstrip('Z')
            data['timestamp'] = datetime.fromisoformat(timestamp_str)
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "LogEntry":
        """
        Create a LogEntry from a JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            LogEntry instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)