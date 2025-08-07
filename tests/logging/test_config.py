"""
Unit tests for LoggerConfig configuration class.

Tests cover configuration validation, defaults, helper methods, and edge cases
to ensure the LoggerConfig class behaves correctly in all scenarios.
"""

import pytest
from unittest.mock import Mock

from src.lmapis.logging.config import LoggerConfig
from src.lmapis.logging.storage import StorageBackend
from src.lmapis.logging.console import Console


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing purposes."""
    
    def save(self, data):
        pass
    
    def close(self):
        pass


class TestLoggerConfigDefaults:
    """Test default configuration values and behavior."""
    
    def test_default_configuration(self):
        """Test that default configuration creates a working setup."""
        config = LoggerConfig()
        
        # Check default values
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.include_request_data is True
        assert config.include_response_data is True
        assert config.include_cost_data is True
        assert config.sanitize_messages is True
        
        # Check default storage backend
        assert len(config.storage_backends) == 1
        assert isinstance(config.storage_backends[0], Console)
        
        # Check that it's properly enabled
        assert config.is_enabled() is True
    
    def test_default_console_backend_format(self):
        """Test that default console backend uses pretty format."""
        config = LoggerConfig()
        console_backend = config.storage_backends[0]
        assert console_backend.format_style == "pretty"
    
    def test_create_console_only_factory_method(self):
        """Test the create_console_only factory method."""
        config = LoggerConfig.create_console_only(format_style="compact")
        
        assert config.enabled is True
        assert len(config.storage_backends) == 1
        assert isinstance(config.storage_backends[0], Console)
        assert config.storage_backends[0].format_style == "compact"
    
    def test_create_disabled_factory_method(self):
        """Test the create_disabled factory method."""
        config = LoggerConfig.create_disabled()
        
        assert config.enabled is False
        assert config.storage_backends == []
        assert config.is_enabled() is False


class TestLoggerConfigValidation:
    """Test configuration parameter validation."""
    
    def test_valid_log_levels(self):
        """Test that valid log levels are accepted and normalized."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            # Test uppercase
            config = LoggerConfig(log_level=level)
            assert config.log_level == level
            
            # Test lowercase (should be normalized to uppercase)
            config = LoggerConfig(log_level=level.lower())
            assert config.log_level == level
            
            # Test mixed case
            config = LoggerConfig(log_level=level.capitalize())
            assert config.log_level == level
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log levels raise ValueError."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            LoggerConfig(log_level="INVALID")
        
        with pytest.raises(ValueError, match="log_level must be one of"):
            LoggerConfig(log_level="TRACE")
    
    def test_storage_backends_must_be_list(self):
        """Test that storage_backends must be a list."""
        with pytest.raises(ValueError, match="storage_backends must be a list"):
            LoggerConfig(storage_backends="not a list")
        
        with pytest.raises(ValueError, match="storage_backends must be a list"):
            LoggerConfig(storage_backends=Console())
    
    def test_storage_backends_cannot_be_empty_when_enabled(self):
        """Test that storage_backends cannot be empty when logging is enabled."""
        with pytest.raises(ValueError, match="storage_backends cannot be empty"):
            LoggerConfig(enabled=True, storage_backends=[])
    
    def test_storage_backends_can_be_empty_when_disabled(self):
        """Test that storage_backends can be empty when logging is disabled."""
        # This should not raise an error because validation is skipped when disabled
        config = LoggerConfig(enabled=False, storage_backends=[])
        assert config.enabled is False
        assert config.storage_backends == []
    
    def test_storage_backends_must_be_storage_backend_instances(self):
        """Test that all items in storage_backends must be StorageBackend instances."""
        with pytest.raises(ValueError, match="must be a StorageBackend instance"):
            LoggerConfig(storage_backends=["not a backend"])
        
        with pytest.raises(ValueError, match="must be a StorageBackend instance"):
            LoggerConfig(storage_backends=[Console(), "invalid", MockStorageBackend()])
    
    def test_boolean_fields_validation(self):
        """Test that boolean fields must be actual booleans."""
        boolean_fields = [
            'enabled', 'include_request_data', 'include_response_data',
            'include_cost_data', 'sanitize_messages'
        ]
        
        for field_name in boolean_fields:
            with pytest.raises(ValueError, match=f"{field_name} must be a boolean"):
                LoggerConfig(**{field_name: "not a boolean"})
            
            with pytest.raises(ValueError, match=f"{field_name} must be a boolean"):
                LoggerConfig(**{field_name: 1})


class TestLoggerConfigHelperMethods:
    """Test helper methods for checking configuration state."""
    
    def test_is_enabled_with_enabled_true_and_backends(self):
        """Test is_enabled returns True when enabled=True and backends exist."""
        config = LoggerConfig(enabled=True, storage_backends=[MockStorageBackend()])
        assert config.is_enabled() is True
    
    def test_is_enabled_with_enabled_false(self):
        """Test is_enabled returns False when enabled=False."""
        config = LoggerConfig(enabled=False, storage_backends=[MockStorageBackend()])
        assert config.is_enabled() is False
    
    def test_is_enabled_with_no_backends(self):
        """Test is_enabled returns False when no storage backends are configured."""
        # This test requires creating a config that bypasses validation
        config = LoggerConfig.create_disabled()
        config.enabled = True  # Manually set to True after creation
        assert config.is_enabled() is False
    
    def test_should_include_data_methods(self):
        """Test the should_include_* helper methods."""
        # Test when enabled and flags are True
        config = LoggerConfig(
            enabled=True,
            include_request_data=True,
            include_response_data=True,
            include_cost_data=True,
            sanitize_messages=True
        )
        
        assert config.should_include_request_data() is True
        assert config.should_include_response_data() is True
        assert config.should_include_cost_data() is True
        assert config.should_sanitize_messages() is True
        
        # Test when enabled but flags are False
        config = LoggerConfig(
            enabled=True,
            include_request_data=False,
            include_response_data=False,
            include_cost_data=False,
            sanitize_messages=False
        )
        
        assert config.should_include_request_data() is False
        assert config.should_include_response_data() is False
        assert config.should_include_cost_data() is False
        assert config.should_sanitize_messages() is False
        
        # Test when disabled (should return False regardless of flags)
        config = LoggerConfig.create_disabled()
        config.include_request_data = True
        config.include_response_data = True
        config.include_cost_data = True
        config.sanitize_messages = True
        
        assert config.should_include_request_data() is False
        assert config.should_include_response_data() is False
        assert config.should_include_cost_data() is False
        assert config.should_sanitize_messages() is False


class TestLoggerConfigBackendManagement:
    """Test methods for managing storage backends."""
    
    def test_add_storage_backend(self):
        """Test adding a storage backend to the configuration."""
        # Start with a disabled config to avoid validation error with empty backends
        config = LoggerConfig.create_disabled()
        mock_backend = MockStorageBackend()
        
        config.add_storage_backend(mock_backend)
        
        assert len(config.storage_backends) == 1
        assert config.storage_backends[0] is mock_backend
    
    def test_add_storage_backend_invalid_type(self):
        """Test that adding an invalid storage backend raises ValueError."""
        config = LoggerConfig()
        
        with pytest.raises(ValueError, match="must be a StorageBackend instance"):
            config.add_storage_backend("not a backend")
    
    def test_remove_storage_backend_by_type(self):
        """Test removing storage backends by type."""
        console1 = Console()
        console2 = Console(format_style="compact")
        mock_backend = MockStorageBackend()
        
        config = LoggerConfig(storage_backends=[console1, mock_backend, console2])
        
        # Remove all Console backends
        removed = config.remove_storage_backend(Console)
        
        assert removed is True
        assert len(config.storage_backends) == 1
        assert config.storage_backends[0] is mock_backend
    
    def test_remove_storage_backend_none_found(self):
        """Test removing storage backends when none of the type exist."""
        config = LoggerConfig(storage_backends=[MockStorageBackend()])
        
        removed = config.remove_storage_backend(Console)
        
        assert removed is False
        assert len(config.storage_backends) == 1
    
    def test_get_storage_backends_by_type(self):
        """Test getting storage backends by type."""
        console1 = Console()
        console2 = Console(format_style="compact")
        mock_backend = MockStorageBackend()
        
        config = LoggerConfig(storage_backends=[console1, mock_backend, console2])
        
        console_backends = config.get_storage_backends_by_type(Console)
        mock_backends = config.get_storage_backends_by_type(MockStorageBackend)
        
        assert len(console_backends) == 2
        assert console1 in console_backends
        assert console2 in console_backends
        
        assert len(mock_backends) == 1
        assert mock_backend in mock_backends
    
    def test_get_storage_backends_by_type_none_found(self):
        """Test getting storage backends when none of the type exist."""
        config = LoggerConfig(storage_backends=[MockStorageBackend()])
        
        console_backends = config.get_storage_backends_by_type(Console)
        
        assert console_backends == []


class TestLoggerConfigRepresentation:
    """Test string representation of LoggerConfig."""
    
    def test_repr_with_default_config(self):
        """Test __repr__ with default configuration."""
        config = LoggerConfig()
        repr_str = repr(config)
        
        assert "LoggerConfig(" in repr_str
        assert "enabled=True" in repr_str
        assert "backends=['Console']" in repr_str
        assert "log_level=INFO" in repr_str
        assert "include_request=True" in repr_str
        assert "include_response=True" in repr_str
        assert "include_cost=True" in repr_str
        assert "sanitize=True" in repr_str
    
    def test_repr_with_multiple_backends(self):
        """Test __repr__ with multiple storage backends."""
        config = LoggerConfig(storage_backends=[Console(), MockStorageBackend()])
        repr_str = repr(config)
        
        assert "backends=['Console', 'MockStorageBackend']" in repr_str
    
    def test_repr_with_disabled_config(self):
        """Test __repr__ with disabled configuration."""
        config = LoggerConfig.create_disabled()
        repr_str = repr(config)
        
        assert "enabled=False" in repr_str
        assert "backends=[]" in repr_str


class TestLoggerConfigEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_multiple_validation_errors(self):
        """Test that validation catches the first error when multiple issues exist."""
        # This should fail on log_level validation first
        with pytest.raises(ValueError, match="log_level must be one of"):
            LoggerConfig(
                log_level="INVALID",
                storage_backends="also invalid",
                enabled="not a boolean"
            )
    
    def test_config_with_mixed_backend_types(self):
        """Test configuration with multiple different backend types."""
        console = Console()
        mock_backend = MockStorageBackend()
        
        config = LoggerConfig(storage_backends=[console, mock_backend])
        
        assert len(config.storage_backends) == 2
        assert config.is_enabled() is True
        
        # Test type-specific operations
        console_backends = config.get_storage_backends_by_type(Console)
        mock_backends = config.get_storage_backends_by_type(MockStorageBackend)
        
        assert len(console_backends) == 1
        assert len(mock_backends) == 1
    
    def test_config_modification_after_creation(self):
        """Test that configuration can be modified after creation."""
        config = LoggerConfig()
        
        # Modify configuration
        config.log_level = "DEBUG"
        config.include_request_data = False
        config.add_storage_backend(MockStorageBackend())
        
        assert config.log_level == "DEBUG"
        assert config.include_request_data is False
        assert len(config.storage_backends) == 2
        
        # Helper methods should reflect changes
        assert config.should_include_request_data() is False