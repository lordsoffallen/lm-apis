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
    
    def log_interaction(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        cost_data: Dict[str, Any],
        error: Optional[Exception] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log a complete LLM interaction including request, response, and cost data.
        
        This method captures all relevant information from an LLM interaction
        in a single log entry. It handles data filtering based on configuration
        and ensures that logging errors don't interrupt LLM operations.
        
        Args:
            request_data: Dictionary containing request information:
                - model: str - The LLM model name
                - backend: str - The backend provider (e.g., 'openai', 'anthropic')
                - messages: List[Dict] - List of message dictionaries
                - parameters: Dict - Request parameters (temperature, max_tokens, etc.)
            response_data: Dictionary containing response information:
                - response_content: str - The generated response content
                - finish_reason: str - Why the model stopped generating
                - tokens_prompt: int - Number of prompt tokens used
                - tokens_completion: int - Number of completion tokens generated
            cost_data: Dictionary containing cost information:
                - cost: float - Total cost of the request in dollars
            error: Optional exception that occurred during the request
            start_time: Optional start timestamp (time.time() format)
            end_time: Optional end timestamp (time.time() format)
            request_id: Optional unique identifier for the request
        """
        if not self.is_enabled():
            return
        
        try:
            # Calculate duration if timestamps provided
            duration_ms = None
            if start_time is not None and end_time is not None:
                duration_ms = int((end_time - start_time) * 1000)
            
            # Create log entry with data filtering based on configuration
            log_entry = LogEntry.create(
                model=request_data.get('model', 'unknown'),
                backend=request_data.get('backend', 'unknown'),
                messages=request_data.get('messages') if self.config.should_include_request_data() else None,
                parameters=request_data.get('parameters') if self.config.should_include_request_data() else None,
                response_content=response_data.get('response_content') if self.config.should_include_response_data() else None,
                finish_reason=response_data.get('finish_reason') if self.config.should_include_response_data() else None,
                cost=cost_data.get('cost') if self.config.should_include_cost_data() else None,
                tokens_prompt=response_data.get('tokens_prompt') if self.config.should_include_cost_data() else None,
                tokens_completion=response_data.get('tokens_completion') if self.config.should_include_cost_data() else None,
                duration_ms=duration_ms,
                error=str(error) if error else None,
                retry_count=0,  # This is for initial requests, retries use log_retry
                request_id=request_id,
            )
            
            # Apply sanitization if configured
            if self.config.should_sanitize_messages():
                log_entry = log_entry.sanitize()
            
            # Save to all configured storage backends
            self._save_to_backends(log_entry.to_dict())
            
        except Exception as e:
            # Log internal error but don't interrupt LLM operations
            self._internal_logger.warning(f"Failed to log LLM interaction: {e}")
    
    def log_retry(
        self,
        attempt: int,
        exception: Exception,
        context: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> None:
        """
        Log a retry attempt with context information.
        
        This method logs information about retry attempts, including the attempt
        number, the exception that triggered the retry, and any relevant context.
        
        Args:
            attempt: The retry attempt number (1-based)
            exception: The exception that triggered the retry
            context: Dictionary containing context information:
                - model: str - The LLM model name
                - backend: str - The backend provider
                - messages: List[Dict] - Original messages (optional)
                - parameters: Dict - Request parameters (optional)
            request_id: Optional unique identifier linking to the original request
        """
        if not self.is_enabled():
            return
        
        try:
            # Create log entry for retry attempt
            log_entry = LogEntry.create(
                model=context.get('model', 'unknown'),
                backend=context.get('backend', 'unknown'),
                messages=context.get('messages') if self.config.should_include_request_data() else None,
                parameters=context.get('parameters') if self.config.should_include_request_data() else None,
                error=f"Retry attempt {attempt}: {str(exception)}",
                retry_count=attempt,
                request_id=request_id or f"retry_{uuid.uuid4().hex[:8]}",
            )
            
            # Apply sanitization if configured
            if self.config.should_sanitize_messages():
                log_entry = log_entry.sanitize()
            
            # Save to all configured storage backends
            self._save_to_backends(log_entry.to_dict())
            
        except Exception as e:
            # Log internal error but don't interrupt retry logic
            self._internal_logger.warning(f"Failed to log retry attempt: {e}")
    
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
