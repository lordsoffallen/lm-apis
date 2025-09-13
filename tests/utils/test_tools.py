from lmapis.utils.tools import Tools, TEXT_EDITOR_TOOL, handle_text_editor_tool, \
    execute_tool, _validate_format
from lmapis.utils.messages import Tool
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall, Function

import json
import pytest


def sample_function(param1: str, param2: int = 10):
    """A sample function for testing.
    
    Args:
        param1: First parameter description
        param2: Second parameter description
    """
    return f"Result: {param1}, {param2}"

def create_mock_tool_call(
    function_name: str, arguments: dict
) -> ChatCompletionMessageToolCall:
    """Helper to create mock tool call."""
    return ChatCompletionMessageToolCall(
        id="call_123",
        function=Function(name=function_name, arguments=json.dumps(arguments)),
        type="function"
    )


class TestToolsFormatSpecification:
    """Test format specification functionality in Tools class."""
    
    def test_init_with_default_format(self):
        """Test Tools initialization with default format."""
        tools = Tools()
        assert tools._format == "openai"
    
    def test_init_with_openai_format(self):
        """Test Tools initialization with openai format."""
        tools = Tools(api_format="openai")
        assert tools._format == "openai"
    
    def test_init_with_anthropic_format(self):
        """Test Tools initialization with anthropic format."""
        tools = Tools(api_format="anthropic")
        assert tools._format == "anthropic"
    
    def test_init_with_invalid_format(self):
        """Test Tools initialization with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format 'invalid'"):
            Tools(api_format="invalid")
    
    def test_validate_format_valid(self):
        """Test format validation with valid formats."""

        assert _validate_format("openai") == "openai"
        assert _validate_format("anthropic") == "anthropic"
    
    def test_validate_format_invalid(self):
        """Test format validation with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format 'invalid'"):
            _validate_format("invalid")
    
    def test_format_method_uses_instance_format_openai(self):
        """Test format() method uses instance format when no format specified (openai)."""
        tools = Tools([sample_function], api_format="openai")
        result = tools.format()
        
        # Should return openai format
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert "function" in result[0]
    
    def test_format_method_uses_instance_format_anthropic(self):
        """Test format() method uses instance format when no format specified (anthropic)."""
        tools = Tools([sample_function], api_format="anthropic")
        result = tools.format()
        
        # Should return anthropic format (raw tool specs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "function" in result[0]  # Contains the function
        assert "spec" in result[0]      # Contains the spec
    
    def test_format_method_with_explicit_format_override(self):
        """Test format() method with explicit format parameter overrides instance format."""
        tools = Tools([sample_function], api_format="anthropic")
        
        # Override with openai format
        result = tools.format(api_format="openai")
        assert result[0]["type"] == "function"
        
        # Override with anthropic format
        result = tools.format(api_format="anthropic")
        assert "function" in result[0]
        assert "spec" in result[0]
    
    def test_format_method_validates_explicit_format(self):
        """Test format() method validates explicit format parameter."""
        tools = Tools([sample_function])
        
        with pytest.raises(ValueError, match="Unsupported format 'invalid'"):
            tools.format(api_format="invalid")
    
    def test_tools_with_dict_tool_specification(self):
        """Test Tools with dict tool specification works with format."""
        dict_tool = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Test parameter"}
                },
                "required": ["param"]
            }
        }
        
        tools = Tools([dict_tool], api_format="openai")
        result = tools.format()
        
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
    
    def test_tools_with_mixed_tool_types(self):
        """Test Tools with mixed function and dict tools."""
        dict_tool = {
            "name": "dict_tool",
            "description": "A dict tool"
        }
        
        tools = Tools([sample_function, dict_tool], api_format="openai")
        result = tools.format()
        
        assert len(result) == 2
        for tool in result:
            assert tool["type"] == "function"
            assert "function" in tool


class TestToolsBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_existing_tools_method_call_still_works(self):
        """Test that calling tools.format(format="openai") still works as before."""
        tools = Tools([sample_function])
        result = tools.format(api_format="openai")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
    
    def test_text_editor_tool_constant_unchanged(self):
        """Test that TEXT_EDITOR_TOOL constant is unchanged."""
        assert TEXT_EDITOR_TOOL == {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool"
        }


class TestToolsFormatMethod:
    """Test the updated format method functionality."""
    
    def test_format_method_with_openai_format(self):
        """Test format method returns OpenAI format correctly."""
        tools = Tools([sample_function])
        result = tools.format("openai")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert "function" in result[0]
        assert result[0]["function"]["name"] == "sample_function"
    
    def test_format_method_with_anthropic_format(self):
        """Test format method returns Anthropic format correctly."""
        tools = Tools([sample_function])
        result = tools.format("anthropic")
        
        assert isinstance(result, list)
        assert len(result) == 1
        # Anthropic format returns the raw tool objects
        assert "function" in result[0]
        assert "spec" in result[0]
    
    def test_format_method_uses_instance_format_when_none(self):
        """Test format method uses instance format when no format specified."""
        tools = Tools([sample_function], api_format="anthropic")
        result = tools.format()  # No format specified
        
        # Should use instance format (anthropic)
        assert isinstance(result, list)
        assert "function" in result[0]
        assert "spec" in result[0]
    
    def test_format_method_validates_format(self):
        """Test format method validates the provided format."""
        tools = Tools([sample_function])
        
        with pytest.raises(ValueError, match="Unsupported format 'invalid'"):
            tools.format("invalid")


class TestTextEditorTool:
    """Test the text editor tool functionality."""

    def test_handle_text_editor_view_command(self):
        """Test text editor view command."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "view"}
        )
        text = "Hello\nWorld"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert result.content == text
        assert result.tool_call_id == "call_123"
    
    def test_handle_text_editor_str_replace_command(self):
        """Test text editor str_replace command."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "str_replace", "old_str": "World", "new_str": "Universe"}
        )
        text = "Hello\nWorld"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert result.content == "Text replaced successfully"
        assert result.call_response == "Hello\nUniverse"
        assert result.tool_call_id == "call_123"
    
    def test_handle_text_editor_str_replace_not_found(self):
        """Test text editor str_replace when string not found."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "str_replace", "old_str": "NotFound", "new_str": "Universe"}
        )
        text = "Hello\nWorld"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert "Error: String not found" in result.content
    
    def test_handle_text_editor_str_replace_multiple_matches(self):
        """Test text editor str_replace with multiple matches."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "str_replace", "old_str": "Hello", "new_str": "Hi"}
        )
        text = "Hello\nHello World"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert "appears 2 times" in result.content
    
    def test_handle_text_editor_str_replace_empty_old_str(self):
        """Test text editor str_replace with empty old_str."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "str_replace", "old_str": "", "new_str": "Hi"}
        )
        text = "Hello World"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert "Error: old_str cannot be empty" in result.content
    
    def test_handle_text_editor_insert_command(self):
        """Test text editor insert command."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "insert", "insert_line": 1, "new_str": "Inserted line"}
        )
        text = "Line 0\nLine 1"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert result.content == "Text inserted at line 1"
        assert result.call_response == "Line 0\nInserted line\nLine 1"
    
    def test_handle_text_editor_insert_invalid_line(self):
        """Test text editor insert with invalid line number."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "insert", "insert_line": 10, "new_str": "Inserted line"}
        )
        text = "Line 0\nLine 1"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert "Error: Invalid line number" in result.content
    
    def test_handle_text_editor_unknown_command(self):
        """Test text editor with unknown command."""
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "unknown_command"}
        )
        text = "Hello World"
        
        result = handle_text_editor_tool(tool_call, text)
        
        assert isinstance(result, Tool)
        assert "Error: Unknown or unsupported command" in result.content
    
    def test_handle_text_editor_wrong_function_name(self):
        """Test text editor with wrong function name."""
        tool_call = create_mock_tool_call(
            "wrong_function",
            {"command": "view"}
        )
        text = "Hello World"
        
        with pytest.raises(ValueError, match="This function is for str based edit tool"):
            handle_text_editor_tool(tool_call, text)


class TestExecuteTool:
    """Test the execute_tool function."""

    def test_execute_tool_with_function_tool(self):
        """Test execute_tool with a regular function tool."""
        tools = Tools([sample_function])
        tool_call = create_mock_tool_call(
            "sample_function",
            {"param1": "test", "param2": 20}
        )
        
        result = execute_tool(tools, tool_call)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Tool)
        assert result[0].tool_call_id == "call_123"
        assert result[0].call_response == "Result: test, 20"
    
    def test_execute_tool_with_text_editor(self):
        """Test execute_tool with text editor tool."""
        tools = Tools([TEXT_EDITOR_TOOL])
        tool_call = create_mock_tool_call(
            TEXT_EDITOR_TOOL["name"],
            {"command": "view"}
        )
        input_files = "Hello World"
        
        result = execute_tool(tools, tool_call, input_files)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Tool)
        assert result[0].content == "Hello World"
    
    def test_execute_tool_with_multiple_tool_calls(self):
        """Test execute_tool with multiple tool calls."""
        tools = Tools([sample_function])
        tool_calls = [
            create_mock_tool_call("sample_function", {"param1": "test1", "param2": 10}),
            create_mock_tool_call("sample_function", {"param1": "test2", "param2": 20})
        ]
        
        result = execute_tool(tools, tool_calls)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, Tool) for r in result)
    
    def test_execute_tool_unregistered_tool(self):
        """Test execute_tool with unregistered tool."""
        tools = Tools([sample_function])
        tool_call = create_mock_tool_call("unregistered_tool", {})
        
        with pytest.raises(ValueError, match="Tool 'unregistered_tool' not registered"):
            execute_tool(tools, tool_call)
    
    def test_execute_tool_text_editor_no_input_files(self):
        """Test execute_tool with text editor but no input files."""
        tools = Tools([TEXT_EDITOR_TOOL])
        tool_call = create_mock_tool_call(TEXT_EDITOR_TOOL["name"], {"command": "view"})
        
        with pytest.raises(ValueError, match="Input files are required for text editor call"):
            execute_tool(tools, tool_call, input_files=None)
    
    def test_execute_tool_text_editor_too_many_files(self):
        """Test execute_tool with text editor and too many input files."""
        tools = Tools([TEXT_EDITOR_TOOL])
        tool_call = create_mock_tool_call(TEXT_EDITOR_TOOL["name"], {"command": "view"})
        input_files = ["file1.txt", "file2.txt", "file3.txt"]
        
        with pytest.raises(ValueError, match="Text Editor only works with a single file"):
            execute_tool(tools, tool_call, input_files)


class TestValidateFormat:
    """Test the _validate_format function."""
    
    def test_validate_format_openai(self):
        """Test _validate_format with openai format."""
        result = _validate_format("openai")
        assert result == "openai"
    
    def test_validate_format_anthropic(self):
        """Test _validate_format with anthropic format."""
        result = _validate_format("anthropic")
        assert result == "anthropic"
    
    def test_validate_format_invalid(self):
        """Test _validate_format with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format 'invalid'"):
            _validate_format("invalid")