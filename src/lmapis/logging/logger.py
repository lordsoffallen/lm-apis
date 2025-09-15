"""
Main LLM logging class for flexible LLM logging system.

This module provides the LLMLogger class that serves as the main interface
for logging LLM interactions. It handles multiple storage backends with
error isolation and graceful degradation.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import LoggerConfig
from .log_entry import LogEntry
from .storage import StorageBackend


class LLMLogger:
    """
    Main logging class for capturing LLM interactions.
    
    This class provides the primary interface for logging complete LLM interactions
    including requests, responses, costs, and errors. It supports multiple storage
    backends with error isolation to ensure that logging failures don't impact
    LLM operations.
    
    Key features:
    - Multiple storage backends with independent error handling
    - Graceful degradation when storage backends fail
    - Complete interaction logging (request + response in single entry)
    - Retry attempt logging
    - Configurable data inclusion and sanitization
    """
    
    def __init__(self, config: LoggerConfig):
        """
        Initialize the LLM logger with the provided configuration.
        
        Args:
            config: LoggerConfig instance defining logging behavior
            
        Raises:
            ValueError: If config is not a LoggerConfig instance
        """
        if not isinstance(config, LoggerConfig):
            raise ValueError(f"config must be a LoggerConfig instance, got: {type(config)}")
        
        self.config = config
        self._closed = False
        
        # Set up Python logging for internal warnings/errors
        self._internal_logger = logging.getLogger(__name__)
        
        # Track failed backends to avoid repeated error messages
        self._failed_backends: set = set()
    
    def is_enabled(self) -> bool:
        """
        Check if logging is enabled and properly configured.
        
        Returns:
            True if logging should occur, False otherwise
        """
        return not self._closed and self.config.is_enabled()
    
    def log_interaction(self, log_entry: LogEntry) -> None:
        """
        Log a complete LLM interaction using a LogEntry object.
        
        This method logs a complete LLM interaction with data filtering based on
        configuration and ensures that logging errors don't interrupt LLM operations.
        
        Args:
            log_entry: LogEntry object containing all interaction data
        """
        if not self.is_enabled():
            return
        
        try:
            # Apply data filtering based on configuration
            filtered_entry = self._apply_data_filtering(log_entry)
            
            # Apply sanitization if configured
            if self.config.should_sanitize_messages():
                filtered_entry = filtered_entry.sanitize()
            
            # Save to all configured storage backends
            self._save_to_backends(filtered_entry.to_dict())
            
        except Exception as e:
            # Log internal error but don't interrupt LLM operations
            self._internal_logger.warning(f"Failed to log LLM interaction: {e}")
    
    def log_retry(self, log_entry: LogEntry) -> None:
        """
        Log a retry attempt using a LogEntry object.
        
        This method logs information about retry attempts with data filtering
        based on configuration and ensures that logging errors don't interrupt
        retry logic.
        
        Args:
            log_entry: LogEntry object containing retry attempt data
        """
        if not self.is_enabled():
            return
        
        try:
            # Apply data filtering based on configuration
            filtered_entry = self._apply_data_filtering(log_entry)
            
            # Apply sanitization if configured
            if self.config.should_sanitize_messages():
                filtered_entry = filtered_entry.sanitize()
            
            # Save to all configured storage backends
            self._save_to_backends(filtered_entry.to_dict())
            
        except Exception as e:
            # Log internal error but don't interrupt retry logic
            self._internal_logger.warning(f"Failed to log retry attempt: {e}")
    
    def _apply_data_filtering(self, log_entry: LogEntry) -> LogEntry:
        """
        Apply data filtering based on configuration settings.
        
        This method creates a new LogEntry with fields filtered out based on
        the logger configuration (e.g., exclude request data, response data, etc.).
        
        Args:
            log_entry: Original LogEntry to filter
            
        Returns:
            New LogEntry with filtered data
        """
        return LogEntry.create(
            model=log_entry.model,
            backend=log_entry.backend,
            messages=log_entry.messages if self.config.should_include_request_data() else None,
            parameters=log_entry.parameters if self.config.should_include_request_data() else None,
            response_content=log_entry.response_content if self.config.should_include_response_data() else None,
            finish_reason=log_entry.finish_reason if self.config.should_include_response_data() else None,
            tool_calls=log_entry.tool_calls if self.config.should_include_response_data() else None,
            cost=log_entry.cost if self.config.should_include_cost_data() else None,
            tokens_prompt=log_entry.tokens_prompt if self.config.should_include_cost_data() else None,
            tokens_completion=log_entry.tokens_completion if self.config.should_include_cost_data() else None,
            duration_ms=log_entry.duration_ms,
            error=log_entry.error,
            retry_count=log_entry.retry_count,
            request_id=log_entry.request_id,
            timestamp=log_entry.timestamp,
        )
    
    def _save_to_backends(self, log_data: Dict[str, Any]) -> None:
        """
        Save log data to all configured storage backends with error isolation.
        
        This method ensures that failures in one storage backend don't affect
        others. Failed backends are tracked to avoid repeated error messages.
        
        Args:
            log_data: Dictionary representation of the log entry
        """
        successful_backends = 0
        
        for backend in self.config.storage_backends:
            try:
                backend.save(log_data)
                successful_backends += 1
                
                # Remove from failed backends set if it was previously failing
                if backend in self._failed_backends:
                    self._failed_backends.remove(backend)
                    self._internal_logger.info(f"Storage backend {type(backend).__name__} recovered")
                    
            except Exception as e:
                # Track failed backend to avoid spam
                if backend not in self._failed_backends:
                    self._failed_backends.add(backend)
                    self._internal_logger.warning(
                        f"Storage backend {type(backend).__name__} failed: {e}. "
                        f"Will suppress further errors from this backend."
                    )
        
        # If all backends failed, log a warning (but don't interrupt operations)
        if successful_backends == 0 and self.config.storage_backends:
            self._internal_logger.error(
                f"All {len(self.config.storage_backends)} storage backends failed. "
                f"Log entry lost: {log_data.get('request_id', 'unknown')}"
            )
    
    def add_storage_backend(self, backend: StorageBackend) -> None:
        """
        Add a storage backend to the logger configuration.
        
        Args:
            backend: StorageBackend instance to add
            
        Raises:
            ValueError: If backend is not a StorageBackend instance
        """
        if not isinstance(backend, StorageBackend):
            raise ValueError(f"backend must be a StorageBackend instance, got: {type(backend)}")
        
        self.config.add_storage_backend(backend)
        
        # Remove from failed backends if it was previously failing
        if backend in self._failed_backends:
            self._failed_backends.remove(backend)
    
    def remove_storage_backend(self, backend_type: type) -> bool:
        """
        Remove all storage backends of a specific type.
        
        Args:
            backend_type: Type of storage backend to remove
            
        Returns:
            True if any backends were removed, False otherwise
        """
        # Remove from failed backends tracking
        backends_to_remove = [
            backend for backend in self._failed_backends
            if isinstance(backend, backend_type)
        ]
        for backend in backends_to_remove:
            self._failed_backends.remove(backend)
        
        return self.config.remove_storage_backend(backend_type)
    
    def get_failed_backends(self) -> List[StorageBackend]:
        """
        Get a list of storage backends that have failed.
        
        Returns:
            List of failed storage backend instances
        """
        return list(self._failed_backends)
    
    def reset_failed_backends(self) -> None:
        """
        Reset the failed backends tracking.
        
        This can be useful for retrying previously failed backends
        after fixing configuration issues.
        """
        self._failed_backends.clear()
        self._internal_logger.info("Reset failed backends tracking")
    
    def close(self) -> None:
        """
        Close the logger and clean up all storage backend resources.
        
        This method should be called when the logger is no longer needed.
        It ensures that all storage backends are properly closed and
        any buffered data is flushed.
        """
        if self._closed:
            return
        
        self._closed = True
        
        # Close all storage backends
        for backend in self.config.storage_backends:
            try:
                backend.close()
            except Exception as e:
                self._internal_logger.warning(f"Error closing storage backend {type(backend).__name__}: {e}")
        
        # Clear failed backends tracking
        self._failed_backends.clear()
    
    def __enter__(self):
        """Context manager entry - returns self for use in with statements."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.close()
    
    def __repr__(self) -> str:
        """Return a string representation of the logger."""
        backend_types = [type(backend).__name__ for backend in self.config.storage_backends]
        failed_count = len(self._failed_backends)
        
        return (
            f"LLMLogger(enabled={self.is_enabled()}, "
            f"backends={backend_types}, "
            f"failed_backends={failed_count}, "
            f"closed={self._closed})"
        )
