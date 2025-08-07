"""
Unit tests for the StorageBackend abstract interface.

These tests validate that the StorageBackend interface is properly defined
and that concrete implementations follow the expected contract.
"""

import pytest
from abc import ABC
from typing import Any, Dict
from unittest.mock import Mock

from lmapis.logging.storage import StorageBackend


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