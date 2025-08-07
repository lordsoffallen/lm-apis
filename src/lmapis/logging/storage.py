"""
Abstract storage backend interface for flexible LLM logging system.

This module defines the StorageBackend abstract base class that provides
a common interface for different storage implementations (console, files, cloud storage).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class StorageBackend(ABC):
    """
    Abstract base class for storage backends in the LLM logging system.
    
    This interface defines the contract that all storage implementations must follow.
    Storage backends are responsible for persisting log entries to their respective
    destinations (console, files, cloud storage, etc.).
    
    The interface is designed to be simple and focused:
    - save(): Store a single log entry
    - close(): Clean up resources when done
    """
    
    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a single log entry to the storage backend.
        
        This method should handle the storage of a complete log entry dictionary.
        Implementations should handle errors gracefully and not raise exceptions
        that would interrupt the main LLM operation flow.
        
        Args:
            data: Dictionary representation of a log entry, typically from
                  LogEntry.to_dict(). Contains all the structured log data
                  including timestamp, request/response data, costs, etc.
                  
        Raises:
            Should not raise exceptions that would interrupt LLM operations.
            Implementations should handle errors internally and optionally
            log warnings to a fallback mechanism.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources and finalize storage operations.
        
        This method is called when the logging system is shutting down or
        when the storage backend is no longer needed. Implementations should:
        - Flush any buffered data
        - Close file handles or network connections
        - Clean up temporary resources
        - Handle cleanup errors gracefully
        
        This method should be idempotent - calling it multiple times should
        not cause errors.
        """
        pass
    
    def __enter__(self):
        """Context manager entry - returns self for use in with statements."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.close()