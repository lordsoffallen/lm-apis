"""
Configuration classes for the flexible LLM logging system.

This module provides the LoggerConfig dataclass that defines how the logging
system should behave, including which storage backends to use and what data
to include in log entries.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .storage import StorageBackend
from .console import Console


@dataclass
class LoggerConfig:
    """
    Configuration class for the LLM logging system.
    
    This dataclass defines all the configuration options for the logging system,
    including which storage backends to use, what data to include, and how to
    handle different scenarios.
    
    The design follows the principle of sensible defaults - a LoggerConfig()
    with no parameters will create a working console-only logging setup.
    """
    
    # Core functionality flags
    enabled: bool = True
    """Whether logging is enabled at all. If False, no logging operations will occur."""
    
    storage_backends: List[StorageBackend] = field(default_factory=lambda: [Console()])
    """List of storage backends to use. Defaults to console-only logging."""
    
    # Log level configuration
    log_level: str = "INFO"
    """Minimum log level to capture. Standard levels: DEBUG, INFO, WARNING, ERROR."""
    
    # Data inclusion flags
    include_request_data: bool = True
    """Whether to include request data (messages, parameters) in log entries."""
    
    include_response_data: bool = True
    """Whether to include response data (content, finish_reason) in log entries."""
    
    include_cost_data: bool = True
    """Whether to include cost and token usage data in log entries."""
    
    # Privacy and security
    sanitize_messages: bool = True
    """Whether to sanitize messages to remove potentially sensitive information."""
    
    def __post_init__(self):
        """
        Validate configuration parameters after initialization.
        
        This method is called automatically by the dataclass after __init__
        and performs validation to ensure the configuration is valid.
        
        Raises:
            ValueError: If any configuration parameters are invalid
        """
        self._validate_log_level()
        self._validate_storage_backends()
        self._validate_boolean_flags()
    
    def _validate_log_level(self) -> None:
        """Validate that log_level is a recognized logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got: {self.log_level}"
            )
        # Normalize to uppercase
        self.log_level = self.log_level.upper()
    
    def _validate_storage_backends(self) -> None:
        """Validate that storage_backends is a list of StorageBackend instances."""
        if not isinstance(self.storage_backends, list):
            raise ValueError("storage_backends must be a list")
        
        # Only require non-empty backends when logging is enabled
        if self.enabled and not self.storage_backends:
            raise ValueError("storage_backends cannot be empty when logging is enabled")
        
        for i, backend in enumerate(self.storage_backends):
            if not isinstance(backend, StorageBackend):
                raise ValueError(
                    f"storage_backends[{i}] must be a StorageBackend instance, "
                    f"got: {type(backend)}"
                )
    
    def _validate_boolean_flags(self) -> None:
        """Validate that all boolean flags are actually boolean values."""
        boolean_fields = [
            'enabled', 'include_request_data', 'include_response_data',
            'include_cost_data', 'sanitize_messages'
        ]
        
        for field_name in boolean_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a boolean, got: {type(value)}")
    
    def is_enabled(self) -> bool:
        """
        Check if logging is enabled and properly configured.
        
        Returns:
            True if logging should occur, False otherwise
        """
        return self.enabled and bool(self.storage_backends)
    
    def should_include_request_data(self) -> bool:
        """Check if request data should be included in log entries."""
        return self.enabled and self.include_request_data
    
    def should_include_response_data(self) -> bool:
        """Check if response data should be included in log entries."""
        return self.enabled and self.include_response_data
    
    def should_include_cost_data(self) -> bool:
        """Check if cost data should be included in log entries."""
        return self.enabled and self.include_cost_data
    
    def should_sanitize_messages(self) -> bool:
        """Check if messages should be sanitized for sensitive information."""
        return self.enabled and self.sanitize_messages
    
    @classmethod
    def create_console_only(cls, format_style: str = "pretty") -> "LoggerConfig":
        """
        Create a configuration with console-only logging.
        
        Args:
            format_style: Console format style ('pretty' or 'compact')
            
        Returns:
            LoggerConfig instance configured for console-only logging
        """
        return cls(storage_backends=[Console(format_style=format_style)])
    
    @classmethod
    def create_disabled(cls) -> "LoggerConfig":
        """
        Create a disabled logging configuration.
        
        Returns:
            LoggerConfig instance with logging disabled
        """
        return cls(enabled=False, storage_backends=[])
    
    def add_storage_backend(self, backend: StorageBackend) -> None:
        """
        Add a storage backend to the configuration.
        
        Args:
            backend: StorageBackend instance to add
            
        Raises:
            ValueError: If backend is not a StorageBackend instance
        """
        if not isinstance(backend, StorageBackend):
            raise ValueError(f"backend must be a StorageBackend instance, got: {type(backend)}")
        
        self.storage_backends.append(backend)
    
    def remove_storage_backend(self, backend_type: type) -> bool:
        """
        Remove all storage backends of a specific type.
        
        Args:
            backend_type: Type of storage backend to remove (e.g., Console, JSONStorage)
            
        Returns:
            True if any backends were removed, False otherwise
        """
        original_count = len(self.storage_backends)
        self.storage_backends = [
            backend for backend in self.storage_backends 
            if not isinstance(backend, backend_type)
        ]
        return len(self.storage_backends) < original_count
    
    def get_storage_backends_by_type(self, backend_type: type) -> List[StorageBackend]:
        """
        Get all storage backends of a specific type.
        
        Args:
            backend_type: Type of storage backend to find
            
        Returns:
            List of storage backends matching the specified type
        """
        return [
            backend for backend in self.storage_backends 
            if isinstance(backend, backend_type)
        ]
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the configuration."""
        backend_types = [type(backend).__name__ for backend in self.storage_backends]
        return (
            f"LoggerConfig(enabled={self.enabled}, "
            f"backends={backend_types}, "
            f"log_level={self.log_level}, "
            f"include_request={self.include_request_data}, "
            f"include_response={self.include_response_data}, "
            f"include_cost={self.include_cost_data}, "
            f"sanitize={self.sanitize_messages})"
        )