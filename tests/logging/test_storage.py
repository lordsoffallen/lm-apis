"""
Unit tests for the StorageBackend abstract interface and JSONStorage implementation.

These tests validate that the StorageBackend interface is properly defined
and that concrete implementations follow the expected contract.
"""

import json
import os
import tempfile
import pytest
from abc import ABC
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock
import warnings

from lmapis.logging.storage import StorageBackend, JSONStorage


class TestStorageBackendInterface:
    """Test the StorageBackend abstract interface definition."""
    
    def test_storage_backend_is_abstract(self):
        """Test that StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            StorageBackend()
    
    def test_storage_backend_inherits_from_abc(self):
        """Test that StorageBackend properly inherits from ABC."""
        assert issubclass(StorageBackend, ABC)
        assert hasattr(StorageBackend, '__abstractmethods__')
        assert 'save' in StorageBackend.__abstractmethods__
        assert 'close' in StorageBackend.__abstractmethods__
    
    def test_abstract_methods_defined(self):
        """Test that the required abstract methods are defined."""
        # Check that save method is abstract
        assert hasattr(StorageBackend, 'save')
        assert getattr(StorageBackend.save, '__isabstractmethod__', False)
        
        # Check that close method is abstract
        assert hasattr(StorageBackend, 'close')
        assert getattr(StorageBackend.close, '__isabstractmethod__', False)
    
    def test_context_manager_methods_exist(self):
        """Test that context manager methods are implemented."""
        assert hasattr(StorageBackend, '__enter__')
        assert hasattr(StorageBackend, '__exit__')
        
        # These should not be abstract since they have default implementations
        assert not getattr(StorageBackend.__enter__, '__isabstractmethod__', False)
        assert not getattr(StorageBackend.__exit__, '__isabstractmethod__', False)


class ConcreteStorageBackend(StorageBackend):
    """Concrete implementation for testing the interface contract."""
    
    def __init__(self):
        self.saved_data = []
        self.is_closed = False
        self.save_call_count = 0
        self.close_call_count = 0
    
    def save(self, data: Dict[str, Any]) -> None:
        """Test implementation that stores data in a list."""
        if self.is_closed:
            raise RuntimeError("Cannot save to closed backend")
        self.saved_data.append(data)
        self.save_call_count += 1
    
    def close(self) -> None:
        """Test implementation that marks the backend as closed."""
        self.is_closed = True
        self.close_call_count += 1


class TestConcreteImplementation:
    """Test that concrete implementations work correctly with the interface."""
    
    def test_concrete_implementation_can_be_instantiated(self):
        """Test that a concrete implementation can be created."""
        backend = ConcreteStorageBackend()
        assert isinstance(backend, StorageBackend)
        assert isinstance(backend, ConcreteStorageBackend)
    
    def test_save_method_works(self):
        """Test that the save method can be called and works correctly."""
        backend = ConcreteStorageBackend()
        test_data = {
            "timestamp": "2024-01-15T10:30:00Z",
            "request_id": "req_test123",
            "model": "gpt-4",
            "backend": "openai"
        }
        
        backend.save(test_data)
        
        assert len(backend.saved_data) == 1
        assert backend.saved_data[0] == test_data
        assert backend.save_call_count == 1
    
    def test_close_method_works(self):
        """Test that the close method can be called and works correctly."""
        backend = ConcreteStorageBackend()
        
        backend.close()
        
        assert backend.is_closed is True
        assert backend.close_call_count == 1
    
    def test_multiple_saves_work(self):
        """Test that multiple save operations work correctly."""
        backend = ConcreteStorageBackend()
        
        data1 = {"request_id": "req_1", "model": "gpt-4"}
        data2 = {"request_id": "req_2", "model": "gpt-3.5"}
        
        backend.save(data1)
        backend.save(data2)
        
        assert len(backend.saved_data) == 2
        assert backend.saved_data[0] == data1
        assert backend.saved_data[1] == data2
        assert backend.save_call_count == 2
    
    def test_close_is_idempotent(self):
        """Test that calling close multiple times doesn't cause issues."""
        backend = ConcreteStorageBackend()
        
        backend.close()
        backend.close()  # Should not raise an error
        
        assert backend.is_closed is True
        assert backend.close_call_count == 2  # Both calls should be recorded
    
    def test_context_manager_functionality(self):
        """Test that the storage backend works as a context manager."""
        with ConcreteStorageBackend() as backend:
            assert isinstance(backend, ConcreteStorageBackend)
            assert not backend.is_closed
            
            test_data = {"request_id": "req_ctx", "model": "gpt-4"}
            backend.save(test_data)
            
            assert len(backend.saved_data) == 1
        
        # After exiting context, close should have been called
        assert backend.is_closed is True
        assert backend.close_call_count == 1
    
    def test_context_manager_calls_close_on_exception(self):
        """Test that close is called even when an exception occurs in the context."""
        backend = None
        
        try:
            with ConcreteStorageBackend() as ctx_backend:
                backend = ctx_backend
                assert not backend.is_closed
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception
        
        # Close should still have been called
        assert backend is not None
        assert backend.is_closed is True
        assert backend.close_call_count == 1


class IncompleteStorageBackend(StorageBackend):
    """Incomplete implementation missing the close method for testing."""
    
    def save(self, data: Dict[str, Any]) -> None:
        pass
    
    # Intentionally missing close() method


class TestInterfaceValidation:
    """Test that the interface properly validates implementations."""
    
    def test_incomplete_implementation_cannot_be_instantiated(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStorageBackend()
    
    def test_mock_implementation_works(self):
        """Test that mock objects can be used for testing."""
        mock_backend = Mock(spec=StorageBackend)
        
        # Mock should have the required abstract methods
        assert hasattr(mock_backend, 'save')
        assert hasattr(mock_backend, 'close')
        
        # Test that we can call the methods
        test_data = {"test": "data"}
        mock_backend.save(test_data)
        mock_backend.close()
        
        # Verify the calls were made
        mock_backend.save.assert_called_once_with(test_data)
        mock_backend.close.assert_called_once()
        
        # Test context manager functionality with manual setup
        mock_backend.__enter__ = Mock(return_value=mock_backend)
        mock_backend.__exit__ = Mock(return_value=None)
        
        with mock_backend as ctx:
            assert ctx is mock_backend
        
        mock_backend.__enter__.assert_called_once()
        mock_backend.__exit__.assert_called_once()


class TestInterfaceDocumentation:
    """Test that the interface has proper documentation."""
    
    def test_class_has_docstring(self):
        """Test that the StorageBackend class has documentation."""
        assert StorageBackend.__doc__ is not None
        assert len(StorageBackend.__doc__.strip()) > 0
        assert "Abstract base class" in StorageBackend.__doc__
    
    def test_save_method_has_docstring(self):
        """Test that the save method has documentation."""
        assert StorageBackend.save.__doc__ is not None
        assert len(StorageBackend.save.__doc__.strip()) > 0
        assert "Save a single log entry" in StorageBackend.save.__doc__
    
    def test_close_method_has_docstring(self):
        """Test that the close method has documentation."""
        assert StorageBackend.close.__doc__ is not None
        assert len(StorageBackend.close.__doc__.strip()) > 0
        assert "Clean up resources" in StorageBackend.close.__doc__


class TestJSONStorageInitialization:
    """Test JSONStorage initialization and configuration."""
    
    def test_json_storage_can_be_instantiated(self):
        """Test that JSONStorage can be created with basic parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            assert isinstance(storage, StorageBackend)
            assert isinstance(storage, JSONStorage)
            assert storage.base_path == temp_dir
            assert storage.partition_by == "date"  # default
            assert not storage.is_closed
    
    def test_json_storage_with_custom_partition(self):
        """Test JSONStorage with different partitioning strategies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test hour partitioning
            storage_hour = JSONStorage(base_path=temp_dir, partition_by="hour")
            assert storage_hour.partition_by == "hour"
            
            # Test no partitioning
            storage_none = JSONStorage(base_path=temp_dir, partition_by="none")
            assert storage_none.partition_by == "none"
    
    def test_json_storage_invalid_partition_raises_error(self):
        """Test that invalid partition_by values raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="partition_by must be"):
                JSONStorage(base_path=temp_dir, partition_by="invalid")
    
    def test_json_storage_with_fs_kwargs(self):
        """Test JSONStorage with filesystem kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fs_kwargs = {"auto_mkdir": True}
            storage = JSONStorage(base_path=temp_dir, fs_kwargs=fs_kwargs)
            assert storage.fs_kwargs == fs_kwargs
    
    def test_json_storage_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from base_path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=f"{temp_dir}/")
            assert storage.base_path == temp_dir
            
            storage2 = JSONStorage(base_path=f"{temp_dir}///")
            assert storage2.base_path == temp_dir


class TestJSONStoragePathGeneration:
    """Test file path generation for different partitioning strategies."""
    
    def test_date_partitioning_path_generation(self):
        """Test path generation with date partitioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            test_time = datetime(2024, 1, 15, 14, 30, 0)
            expected_path = f"{temp_dir}/2024-01-15/2024-01-15.jsonl"
            
            actual_path = storage._generate_file_path(test_time)
            assert actual_path == expected_path
    
    def test_hour_partitioning_path_generation(self):
        """Test path generation with hour partitioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="hour")
            
            test_time = datetime(2024, 1, 15, 14, 30, 0)
            expected_path = f"{temp_dir}/2024-01-15/14/2024-01-15_14.jsonl"
            
            actual_path = storage._generate_file_path(test_time)
            assert actual_path == expected_path
    
    def test_no_partitioning_path_generation(self):
        """Test path generation with no partitioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            test_time = datetime(2024, 1, 15, 14, 30, 0)
            expected_path = f"{temp_dir}/logs.jsonl"
            
            actual_path = storage._generate_file_path(test_time)
            assert actual_path == expected_path
    
    def test_protocol_detection(self):
        """Test protocol detection from base_path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Local file protocol
            storage_local = JSONStorage(base_path=temp_dir)
            assert storage_local._get_protocol() == "file"
            
            # S3 protocol
            storage_s3 = JSONStorage(base_path="s3://bucket/path")
            assert storage_s3._get_protocol() == "s3"
            
            # GCS protocol
            storage_gcs = JSONStorage(base_path="gcs://bucket/path")
            assert storage_gcs._get_protocol() == "gcs"


class TestJSONStorageSaveOperation:
    """Test the save operation with different scenarios."""
    
    def test_save_single_log_entry(self):
        """Test saving a single log entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            test_data = {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_test123",
                "model": "gpt-4",
                "backend": "openai",
                "cost": 0.002
            }
            
            storage.save(test_data)
            
            # Check that file was created
            expected_file = f"{temp_dir}/logs.jsonl"
            assert os.path.exists(expected_file)
            
            # Check file content
            with open(expected_file, 'r') as f:
                content = f.read().strip()
                loaded_data = json.loads(content)
                assert loaded_data == test_data
    
    def test_save_multiple_log_entries(self):
        """Test saving multiple log entries to the same file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            test_data1 = {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_1",
                "model": "gpt-4"
            }
            test_data2 = {
                "timestamp": "2024-01-15T10:31:00Z",
                "request_id": "req_2",
                "model": "gpt-3.5"
            }
            
            storage.save(test_data1)
            storage.save(test_data2)
            
            # Check file content
            expected_file = f"{temp_dir}/logs.jsonl"
            with open(expected_file, 'r') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 2
                assert json.loads(lines[0]) == test_data1
                assert json.loads(lines[1]) == test_data2
    
    def test_save_with_date_partitioning(self):
        """Test saving with date partitioning creates correct directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            test_data = {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_test123",
                "model": "gpt-4"
            }
            
            storage.save(test_data)
            
            # Check directory structure
            expected_dir = f"{temp_dir}/2024-01-15"
            expected_file = f"{expected_dir}/2024-01-15.jsonl"
            
            assert os.path.exists(expected_dir)
            assert os.path.isdir(expected_dir)
            assert os.path.exists(expected_file)
            
            # Check file content
            with open(expected_file, 'r') as f:
                content = f.read().strip()
                loaded_data = json.loads(content)
                assert loaded_data == test_data
    
    def test_save_with_hour_partitioning(self):
        """Test saving with hour partitioning creates correct directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="hour")
            
            test_data = {
                "timestamp": "2024-01-15T14:30:00Z",
                "request_id": "req_test123",
                "model": "gpt-4"
            }
            
            storage.save(test_data)
            
            # Check directory structure
            expected_dir = f"{temp_dir}/2024-01-15/14"
            expected_file = f"{expected_dir}/2024-01-15_14.jsonl"
            
            assert os.path.exists(expected_dir)
            assert os.path.isdir(expected_dir)
            assert os.path.exists(expected_file)
    
    def test_save_with_datetime_object_timestamp(self):
        """Test saving when timestamp is a datetime object instead of string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            test_data = {
                "timestamp": datetime(2024, 1, 15, 10, 30, 0),
                "request_id": "req_test123",
                "model": "gpt-4"
            }
            
            storage.save(test_data)
            
            # Should still work and create correct path
            expected_file = f"{temp_dir}/2024-01-15/2024-01-15.jsonl"
            assert os.path.exists(expected_file)
    
    def test_save_with_missing_timestamp_uses_current_time(self):
        """Test that missing timestamp falls back to current time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            test_data = {
                "request_id": "req_test123",
                "model": "gpt-4"
                # No timestamp field
            }
            
            # Save the data - it should use current time
            storage.save(test_data)
            
            # Check that a file was created (we can't predict exact date, but should exist)
            files = storage.list_files()
            assert len(files) == 1
            assert files[0].endswith(".jsonl")
            
            # Verify the file contains our data
            with open(files[0], 'r') as f:
                content = f.read().strip()
                loaded_data = json.loads(content)
                assert loaded_data["request_id"] == "req_test123"
                assert loaded_data["model"] == "gpt-4"


class TestJSONStorageErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_save_to_closed_storage_shows_warning(self):
        """Test that saving to closed storage shows warning but doesn't crash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            storage.close()
            
            test_data = {"request_id": "req_test", "model": "gpt-4"}
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                storage.save(test_data)
                
                assert len(w) == 1
                assert "closed JSONStorage" in str(w[0].message)
    
    def test_save_with_unserializable_data_converts_to_string(self):
        """Test that unserializable data is converted to string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            # Create data that can't be JSON serialized normally
            class UnserializableClass:
                def __str__(self):
                    return "UnserializableClass instance"
            
            test_data = {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_test",
                "unserializable": UnserializableClass()
            }
            
            # Should not raise an exception or warning - should convert to string
            storage.save(test_data)
            
            # Verify the file was created and contains string representation
            log_file = f"{temp_dir}/logs.jsonl"
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                content = f.read().strip()
                loaded_data = json.loads(content)
                assert loaded_data["request_id"] == "req_test"
                assert "UnserializableClass instance" in loaded_data["unserializable"]
    
    def test_save_with_permission_error_shows_warning(self):
        """Test that permission errors show warning but don't crash."""
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only
            
            try:
                storage = JSONStorage(base_path=readonly_dir)
                test_data = {"request_id": "req_test", "timestamp": "2024-01-15T10:30:00Z"}
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    storage.save(test_data)
                    
                    # Should have at least one warning
                    assert len(w) >= 1
                    warning_messages = [str(warning.message) for warning in w]
                    assert any("Failed to save log entry" in msg for msg in warning_messages)
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)
    
    @patch('lmapis.logging.storage.fsspec.filesystem')
    def test_filesystem_initialization_fallback(self, mock_filesystem):
        """Test fallback to local filesystem when initialization fails."""
        # Make the first call fail, second call succeed
        mock_filesystem.side_effect = [Exception("Connection failed"), MagicMock()]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                storage = JSONStorage(base_path=temp_dir)
                
                # Should have warning about fallback
                assert len(w) == 1
                assert "Failed to initialize filesystem" in str(w[0].message)
                assert "Falling back to local filesystem" in str(w[0].message)
                
                # Should still be able to save
                test_data = {"request_id": "req_test", "timestamp": "2024-01-15T10:30:00Z"}
                storage.save(test_data)  # Should not raise exception


class TestJSONStorageCloseOperation:
    """Test the close operation and resource cleanup."""
    
    def test_close_marks_storage_as_closed(self):
        """Test that close() marks the storage as closed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            assert not storage.is_closed
            
            storage.close()
            assert storage.is_closed
    
    def test_close_is_idempotent(self):
        """Test that calling close() multiple times is safe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            
            storage.close()
            assert storage.is_closed
            
            # Should not raise exception
            storage.close()
            assert storage.is_closed
    
    def test_context_manager_calls_close(self):
        """Test that using JSONStorage as context manager calls close."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with JSONStorage(base_path=temp_dir) as storage:
                assert not storage.is_closed
                
                test_data = {"request_id": "req_test", "timestamp": "2024-01-15T10:30:00Z"}
                storage.save(test_data)
            
            # After exiting context, should be closed
            assert storage.is_closed


class TestJSONStorageFileListingUtilities:
    """Test utility methods for file listing and management."""
    
    def test_list_files_with_no_partitioning(self):
        """Test listing files when no partitioning is used."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            # Initially no files
            files = storage.list_files()
            assert files == []
            
            # Save a log entry
            test_data = {"request_id": "req_test", "timestamp": "2024-01-15T10:30:00Z"}
            storage.save(test_data)
            
            # Now should list the file
            files = storage.list_files()
            assert len(files) == 1
            assert files[0].endswith("logs.jsonl")
    
    def test_list_files_with_date_partitioning(self):
        """Test listing files with date partitioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            # Save entries for different dates
            test_data1 = {"request_id": "req_1", "timestamp": "2024-01-15T10:30:00Z"}
            test_data2 = {"request_id": "req_2", "timestamp": "2024-01-16T10:30:00Z"}
            
            storage.save(test_data1)
            storage.save(test_data2)
            
            files = storage.list_files()
            assert len(files) == 2
            
            # Check that both date directories are represented
            file_paths = [str(f) for f in files]
            assert any("2024-01-15" in path for path in file_paths)
            assert any("2024-01-16" in path for path in file_paths)
    
    def test_list_files_with_date_range_filter(self):
        """Test listing files with date range filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            # Save entries for different dates
            dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
            for i, date in enumerate(dates):
                test_data = {"request_id": f"req_{i}", "timestamp": f"{date}T10:30:00Z"}
                storage.save(test_data)
            
            # Filter for middle date only
            start_date = datetime(2024, 1, 16)
            end_date = datetime(2024, 1, 16)
            
            filtered_files = storage.list_files(start_date=start_date, end_date=end_date)
            assert len(filtered_files) == 1
            assert "2024-01-16" in str(filtered_files[0])
    
    def test_list_files_on_closed_storage(self):
        """Test that listing files on closed storage returns empty list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            storage.close()
            
            files = storage.list_files()
            assert files == []
    
    def test_extract_date_from_path(self):
        """Test date extraction from file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            
            # Test valid date path
            path_with_date = "/logs/2024-01-15/2024-01-15.jsonl"
            extracted_date = storage._extract_date_from_path(path_with_date)
            assert extracted_date == datetime(2024, 1, 15).date()
            
            # Test path without date
            path_without_date = "/logs/logs.jsonl"
            extracted_date = storage._extract_date_from_path(path_without_date)
            assert extracted_date is None


class TestJSONStorageIntegration:
    """Integration tests for JSONStorage with real file operations."""
    
    def test_complete_logging_workflow(self):
        """Test a complete logging workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="date")
            
            # Simulate multiple log entries over time
            log_entries = [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_1",
                    "model": "gpt-4",
                    "backend": "openai",
                    "cost": 0.002,
                    "tokens_prompt": 25,
                    "tokens_completion": 12
                },
                {
                    "timestamp": "2024-01-15T11:45:00Z",
                    "request_id": "req_2",
                    "model": "gpt-3.5-turbo",
                    "backend": "openai",
                    "cost": 0.001,
                    "tokens_prompt": 30,
                    "tokens_completion": 15
                },
                {
                    "timestamp": "2024-01-16T09:15:00Z",
                    "request_id": "req_3",
                    "model": "claude-3",
                    "backend": "anthropic",
                    "cost": 0.003,
                    "tokens_prompt": 40,
                    "tokens_completion": 20
                }
            ]
            
            # Save all entries
            for entry in log_entries:
                storage.save(entry)
            
            # Verify file structure
            files = storage.list_files()
            assert len(files) == 2  # Two different dates
            
            # Verify content of first day's file
            day1_file = f"{temp_dir}/2024-01-15/2024-01-15.jsonl"
            assert os.path.exists(day1_file)
            
            with open(day1_file, 'r') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 2  # Two entries for 2024-01-15
                
                entry1 = json.loads(lines[0])
                entry2 = json.loads(lines[1])
                
                assert entry1["request_id"] == "req_1"
                assert entry2["request_id"] == "req_2"
            
            # Verify content of second day's file
            day2_file = f"{temp_dir}/2024-01-16/2024-01-16.jsonl"
            assert os.path.exists(day2_file)
            
            with open(day2_file, 'r') as f:
                content = f.read().strip()
                entry3 = json.loads(content)
                assert entry3["request_id"] == "req_3"
            
            # Clean up
            storage.close()
            assert storage.is_closed
    
    def test_concurrent_access_simulation(self):
        """Test behavior with simulated concurrent access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir, partition_by="none")
            
            # Simulate multiple rapid saves (like concurrent requests)
            entries = []
            for i in range(10):
                entry = {
                    "timestamp": f"2024-01-15T10:30:{i:02d}Z",
                    "request_id": f"req_{i}",
                    "model": "gpt-4"
                }
                entries.append(entry)
                storage.save(entry)
            
            # Verify all entries were saved
            log_file = f"{temp_dir}/logs.jsonl"
            with open(log_file, 'r') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 10
                
                for i, line in enumerate(lines):
                    saved_entry = json.loads(line)
                    assert saved_entry["request_id"] == f"req_{i}"
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content in log entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JSONStorage(base_path=temp_dir)
            
            test_data = {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_unicode",
                "model": "gpt-4",
                "response_content": "Hello! 你好! こんにちは! 🌟",
                "messages": [
                    {"role": "user", "content": "Translate: café, naïve, résumé"}
                ]
            }
            
            storage.save(test_data)
            
            # Verify Unicode content is preserved
            log_file = f"{temp_dir}/2024-01-15/2024-01-15.jsonl"
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                loaded_data = json.loads(content)
                
                assert loaded_data["response_content"] == "Hello! 你好! こんにちは! 🌟"
                assert loaded_data["messages"][0]["content"] == "Translate: café, naïve, résumé"