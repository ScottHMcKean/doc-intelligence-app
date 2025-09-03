"""Unit tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.config import DocConfig, DotDict, load_config


class TestDotDict:
    """Test the DotDict class."""

    def test_dot_dict_creation(self):
        """Test DotDict creation."""
        data = {"key1": "value1", "key2": {"nested": "value"}}
        dot_dict = DotDict(data)

        assert dot_dict["key1"] == "value1"
        assert dot_dict.key1 == "value1"
        assert dot_dict["key2"]["nested"] == "value"
        assert dot_dict.key2.nested == "value"

    def test_dot_dict_nested_access(self):
        """Test nested access with DotDict."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}
        dot_dict = DotDict(data)

        assert dot_dict.level1.level2.level3 == "deep_value"
        assert dot_dict["level1"]["level2"]["level3"] == "deep_value"

    def test_dot_dict_attribute_error(self):
        """Test DotDict attribute error handling."""
        dot_dict = DotDict({"key1": "value1"})

        with pytest.raises(AttributeError):
            _ = dot_dict.nonexistent_key

    def test_dot_dict_get_nested(self):
        """Test get_nested method."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}
        dot_dict = DotDict(data)

        # Test successful nested access
        assert dot_dict.get_nested("level1.level2.level3") == "deep_value"
        assert dot_dict.get_nested("level1.level2") == {"level3": "deep_value"}
        assert dot_dict.get_nested("level1") == {"level2": {"level3": "deep_value"}}

        # Test with default values
        assert dot_dict.get_nested("nonexistent.key", "default") == "default"
        assert dot_dict.get_nested("level1.nonexistent", "default") == "default"

        # Test with None default
        assert dot_dict.get_nested("nonexistent.key") is None

    def test_dot_dict_get_nested_with_invalid_path(self):
        """Test get_nested with invalid path."""
        data = {"key1": "value1"}
        dot_dict = DotDict(data)

        # Test with invalid path
        assert dot_dict.get_nested("key1.invalid.nested") is None
        assert dot_dict.get_nested("key1.invalid.nested", "default") == "default"


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_success(self, test_config_data):
        """Test successful config loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config == test_config_data
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            os.unlink(config_path)


class TestDocConfig:
    """Test the DocConfig class."""

    def test_doc_config_initialization(self, test_config_data):
        """Test DocConfig initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test basic access
            assert config["application"]["name"] == "Test Document Intelligence"
            assert config.application.name == "Test Document Intelligence"

            # Test nested access
            assert config["storage"]["max_file_size_mb"] == 10
            assert config.storage.max_file_size_mb == 10

        finally:
            os.unlink(config_path)

    def test_doc_config_relative_path(self, test_config_data):
        """Test DocConfig with relative path."""
        # Create config file in current directory
        config_path = "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(test_config_data, f)

        try:
            # Test with relative path
            config = DocConfig("test_config.yaml")
            assert config.application.name == "Test Document Intelligence"
        finally:
            os.unlink(config_path)

    def test_doc_config_absolute_path(self, test_config_data):
        """Test DocConfig with absolute path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            # Test with absolute path
            config = DocConfig(config_path)
            assert config.application.name == "Test Document Intelligence"
        finally:
            os.unlink(config_path)

    def test_doc_config_get_method(self, test_config_data):
        """Test DocConfig get method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test get method with dot notation
            assert config.get("application.name") == "Test Document Intelligence"
            assert config.get("storage.max_file_size_mb") == 10
            assert config.get("agent.llm.max_tokens") == 256

            # Test get method with default values
            assert config.get("nonexistent.key", "default") == "default"
            assert config.get("application.nonexistent", "default") == "default"

            # Test get method with None default
            assert config.get("nonexistent.key") is None

        finally:
            os.unlink(config_path)

    def test_doc_config_reload(self, test_config_data):
        """Test DocConfig reload functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)
            original_name = config.application.name

            # Modify the config file
            modified_data = test_config_data.copy()
            modified_data["application"]["name"] = "Modified Name"

            with open(config_path, "w") as f:
                yaml.dump(modified_data, f)

            # Reload config
            config.reload_config()

            # Check that config was reloaded
            assert config.application.name == "Modified Name"
            assert config.application.name != original_name

        finally:
            os.unlink(config_path)

    def test_doc_config_nested_access(self, test_config_data):
        """Test nested access patterns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test various access patterns
            assert config.application.name == "Test Document Intelligence"
            assert config.application.debug_mode is True
            assert config.storage.max_file_size_mb == 10
            assert config.storage.allowed_extensions == [".pdf", ".txt", ".md"]
            assert config.agent.llm.max_tokens == 256
            assert config.agent.llm.temperature == 0.1
            assert config.agent.retrieval.similarity_threshold == 0.7

        finally:
            os.unlink(config_path)

    def test_doc_config_file_not_found(self):
        """Test DocConfig with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DocConfig("nonexistent_config.yaml")

    def test_doc_config_invalid_yaml(self):
        """Test DocConfig with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                DocConfig(config_path)
        finally:
            os.unlink(config_path)

    def test_doc_config_complex_nested_structure(self):
        """Test DocConfig with complex nested structure."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "deep_value",
                        "list": [1, 2, 3],
                        "nested_dict": {"key": "value"},
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(complex_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test deep nested access
            assert config.level1.level2.level3.level4 == "deep_value"
            assert config.level1.level2.level3.list == [1, 2, 3]
            assert config.level1.level2.level3.nested_dict.key == "value"

            # Test get method with deep nesting
            assert config.get("level1.level2.level3.level4") == "deep_value"
            assert config.get("level1.level2.level3.list") == [1, 2, 3]

        finally:
            os.unlink(config_path)

    def test_doc_config_type_preservation(self, test_config_data):
        """Test that data types are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test type preservation
            assert isinstance(config.application.debug_mode, bool)
            assert isinstance(config.storage.max_file_size_mb, int)
            assert isinstance(config.storage.allowed_extensions, list)
            assert isinstance(config.agent.llm.temperature, float)
            assert isinstance(config.application.name, str)

        finally:
            os.unlink(config_path)

    def test_doc_config_empty_config(self):
        """Test DocConfig with empty configuration."""
        empty_data = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(empty_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test with empty config
            assert config.get("nonexistent.key", "default") == "default"
            assert config.get("nonexistent.key") is None

        finally:
            os.unlink(config_path)

    def test_doc_config_special_characters(self):
        """Test DocConfig with special characters in values."""
        special_data = {
            "special": {
                "unicode": "café",
                "quotes": "value with 'quotes' and \"double quotes\"",
                "newlines": "line1\nline2\nline3",
                "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(special_data, f)
            config_path = f.name

        try:
            config = DocConfig(config_path)

            # Test special characters
            assert config.special.unicode == "café"
            assert config.special.quotes == "value with 'quotes' and \"double quotes\""
            assert config.special.newlines == "line1\nline2\nline3"
            assert config.special.special_chars == "!@#$%^&*()_+-=[]{}|;':\",./<>?"

        finally:
            os.unlink(config_path)
