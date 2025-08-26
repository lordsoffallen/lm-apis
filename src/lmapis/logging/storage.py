"""
Abstract storage backend interface for flexible LLM logging system.

This module defines the StorageBackend abstract base class that provides
a common interface for different storage implementations (console, files, cloud storage).
"""

import json
import os
import fsspec

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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


class JSONStorage(StorageBackend):
    """
    JSON file storage backend using fsspec for filesystem abstraction.
    
    This storage backend writes log entries as JSON files using fsspec,
    which provides a unified interface for local files, S3, GCS, Azure Blob, etc.
    Supports file partitioning by date or hour for better organization.
    """
    
    def __init__(
        self,
        base_path: str,
        partition_by: str = "date",
        fs_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JSONStorage with fsspec filesystem.
        
        Args:
            base_path: Base path for storing files (e.g., "./logs", "s3://bucket/logs")
            partition_by: Partitioning strategy - "date", "hour", or "none"
            fs_kwargs: Additional keyword arguments for fsspec filesystem
        """
        self.base_path = base_path.rstrip('/')
        self.partition_by = partition_by
        self.fs_kwargs = fs_kwargs or {}
        self._fs = None
        self._closed = False
        
        # Validate partition_by parameter
        if partition_by not in ["date", "hour", "none"]:
            raise ValueError(f"partition_by must be 'date', 'hour', or 'none', got: {partition_by}")
        
        # Initialize filesystem
        self._initialize_filesystem()
    
    def _initialize_filesystem(self) -> None:
        """Initialize the fsspec filesystem based on the base_path."""
        try:
            # fsspec.open automatically detects the protocol from the path
            self._fs = fsspec.filesystem(self._get_protocol(), **self.fs_kwargs)
        except Exception as e:
            # Fallback to local filesystem if protocol detection fails
            import warnings
            warnings.warn(f"Failed to initialize filesystem for {self.base_path}: {e}. Falling back to local filesystem.")
            self._fs = fsspec.filesystem('file')
    
    def _get_protocol(self) -> str:
        """Extract the protocol from the base_path."""
        if '://' in self.base_path:
            return self.base_path.split('://')[0]
        return 'file'
    
    def _generate_file_path(self, timestamp: datetime) -> str:
        """
        Generate the file path based on partitioning strategy.
        
        Args:
            timestamp: Timestamp from the log entry
            
        Returns:
            Full file path for the log entry
        """
        if self.partition_by == "date":
            date_str = timestamp.strftime("%Y-%m-%d")
            return f"{self.base_path}/{date_str}/{date_str}.jsonl"
        elif self.partition_by == "hour":
            datetime_str = timestamp.strftime("%Y-%m-%d/%H")
            hour_str = timestamp.strftime("%Y-%m-%d_%H")
            return f"{self.base_path}/{datetime_str}/{hour_str}.jsonl"
        else:  # partition_by == "none"
            return f"{self.base_path}/logs.jsonl"
    
    def _ensure_directory_exists(self, file_path: str) -> None:
        """
        Ensure the directory for the file path exists.
        
        Args:
            file_path: Full file path
        """
        try:
            # Extract directory from file path
            if '/' in file_path:
                directory = '/'.join(file_path.split('/')[:-1])
                if directory and not self._fs.exists(directory):
                    self._fs.makedirs(directory, exist_ok=True)
        except Exception as e:
            # Log warning but don't fail - let the file write attempt handle the error
            import warnings
            warnings.warn(f"Failed to create directory for {file_path}: {e}")
    
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a log entry as a JSON line in the appropriate file.
        
        Args:
            data: Dictionary representation of a log entry
        """
        if self._closed:
            import warnings
            warnings.warn("Attempted to save to closed JSONStorage backend")
            return
        
        try:
            # Parse timestamp from data
            timestamp_str = data.get('timestamp')
            if isinstance(timestamp_str, str):
                # Remove 'Z' suffix if present and parse
                timestamp_str = timestamp_str.rstrip('Z')
                timestamp = datetime.fromisoformat(timestamp_str)
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
            else:
                # Fallback to current time if timestamp is missing or invalid
                timestamp = datetime.utcnow()
            
            # Generate file path based on partitioning strategy
            file_path = self._generate_file_path(timestamp)
            
            # Ensure directory exists
            self._ensure_directory_exists(file_path)
            
            # Convert data to JSON line, handling datetime objects
            json_line = json.dumps(data, ensure_ascii=False, default=self._json_serializer) + '\n'
            
            # Write to file (append mode)
            with self._fs.open(file_path, 'a', encoding='utf-8') as f:
                f.write(json_line)
                
        except Exception as e:
            # Handle errors gracefully - don't interrupt LLM operations
            import warnings
            warnings.warn(f"Failed to save log entry to JSONStorage: {e}")
    
    def close(self) -> None:
        """
        Clean up resources and mark the storage as closed.
        
        This method is idempotent and can be called multiple times safely.
        """
        if not self._closed:
            self._closed = True
            # fsspec filesystems typically don't need explicit cleanup
            # but we mark as closed to prevent further operations
    
    @property
    def is_closed(self) -> bool:
        """Check if the storage backend is closed."""
        return self._closed
    
    def list_files(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> list:
        """
        List log files within a date range (useful for debugging/maintenance).
        
        Args:
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            
        Returns:
            List of file paths matching the criteria
        """
        if self._closed:
            return []
        
        try:
            if self.partition_by == "none":
                # Single file case
                file_path = f"{self.base_path}/logs.jsonl"
                return [file_path] if self._fs.exists(file_path) else []
            
            # For partitioned files, we need to scan directories
            all_files = []
            
            if self.partition_by == "date":
                pattern = f"{self.base_path}/*/*.jsonl"
            else:  # hour
                pattern = f"{self.base_path}/*/*/*.jsonl"
            
            try:
                all_files = self._fs.glob(pattern)
            except Exception:
                # If glob fails, try to list directories manually
                if self._fs.exists(self.base_path):
                    all_files = self._find_files_recursive(self.base_path)
            
            # Filter by date range if provided
            if start_date or end_date:
                filtered_files = []
                for file_path in all_files:
                    file_date = self._extract_date_from_path(file_path)
                    if file_date:
                        if start_date and file_date < start_date.date():
                            continue
                        if end_date and file_date > end_date.date():
                            continue
                    filtered_files.append(file_path)
                return filtered_files
            
            return all_files
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to list files in JSONStorage: {e}")
            return []
    
    def _find_files_recursive(self, path: str) -> list:
        """Recursively find .jsonl files in the given path."""
        files = []
        try:
            for item in self._fs.listdir(path):
                item_path = f"{path}/{item['name']}" if isinstance(item, dict) else f"{path}/{item}"
                if self._fs.isdir(item_path):
                    files.extend(self._find_files_recursive(item_path))
                elif item_path.endswith('.jsonl'):
                    files.append(item_path)
        except Exception:
            pass  # Ignore errors in recursive search
        return files
    
    def _extract_date_from_path(self, file_path: str) -> Optional[datetime.date]:
        """Extract date from file path for filtering."""
        try:
            import re
            # Look for date patterns in the path (YYYY-MM-DD)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
            if date_match:
                return datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
        except Exception:
            pass
        return None
    
    def _json_serializer(self, obj: Any) -> str:
        """
        Custom JSON serializer for handling datetime objects and other non-serializable types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat() + 'Z'
        # For other non-serializable objects, convert to string
        return str(obj)