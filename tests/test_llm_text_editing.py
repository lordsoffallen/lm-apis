"""
Test suite for LLM text editing functionality.

This module tests the LLM class's ability to handle text editing through tool calls,
including single-turn and multi-turn editing scenarios without mocking the 
handle_text_editor_tool function to reflect real-world usage.
"""

import pytest
import json
from unittest.mock import Mock, patch
from lmapis.llm import LLM
from lmapis.utils.messages import Messages, User, Assistant
from lmapis.utils.tools import Tools, TEXT_EDITOR_TOOL
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall, 
    Function
)


class TestLLMTextEditing:
    """Test suite for LLM text editing functionality."""
    
    @pytest.fixture
    def mock_backend_setup(self):
        """Setup mock backend for text editing tests."""
        mock_client = Mock()
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_instance.client = mock_client
        mock_backend_class.return_value = mock_backend_instance
        return mock_backend_class, mock_client
    
    @pytest.fixture
    def text_editor_tools(self):
        """Create tools with text editor for testing."""
        return Tools([TEXT_EDITOR_TOOL], api_format="anthropic")
    
    @staticmethod
    def create_tool_call(tool_name: str, arguments: dict, call_id: str = "call_1"):
        """Create a ChatCompletionMessageToolCall object."""
        return ChatCompletionMessageToolCall(
            id=call_id,
            function=Function(name=tool_name, arguments=json.dumps(arguments)),
            type="function"
        )
    
    @staticmethod
    def create_mock_response(content: str, tool_calls=None, finish_reason="stop"):
        """Create a mock response with optional tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = finish_reason
        mock_response.choices[0].message.tool_calls = tool_calls
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        return mock_response
    
    def test_single_turn_text_replacement(self, mock_backend_setup, text_editor_tools):
        """Test single-turn text editing with str_replace command."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tool call for text replacement
        tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "Hello",
                "new_str": "Hi"
            }
        )
        
        # First response with tool call
        first_response = self.create_mock_response(
            "I'll replace 'Hello' with 'Hi' for you.",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        
        # Second response after tool execution
        final_response = self.create_mock_response(
            "I've successfully replaced 'Hello' with 'Hi' in your text."
        )
        
        mock_client.chat.completions.create.side_effect = [first_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("I'll replace 'Hello' with 'Hi' for you.")
            first_assistant.tool_calls = [tool_call]
            
            # Mock final assistant response
            final_assistant = Assistant("I've successfully replaced 'Hello' with 'Hi' in your text.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input text
            messages = Messages() >> User("Replace 'Hello' with 'Hi' in the text")
            messages.input_text_files = "Hello world! How are you?"
            
            # Execute with max_turns=1 to allow tool execution
            result = llm(messages=messages, tools=text_editor_tools, max_turns=1)
            
            # Verify the result
            assert result == final_assistant
            assert result.text_files == "Hi world! How are you?"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_single_turn_text_insertion(self, mock_backend_setup, text_editor_tools):
        """Test single-turn text editing with insert command."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tool call for text insertion
        tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "insert",
                "insert_line": 1,
                "new_str": "This is a new line."
            }
        )
        
        # First response with tool call
        first_response = self.create_mock_response(
            "I'll insert a new line for you.",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        
        # Second response after tool execution
        final_response = self.create_mock_response(
            "I've successfully inserted the new line."
        )
        
        mock_client.chat.completions.create.side_effect = [first_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("I'll insert a new line for you.")
            first_assistant.tool_calls = [tool_call]
            
            # Mock final assistant response
            final_assistant = Assistant("I've successfully inserted the new line.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input text
            messages = Messages() >> User("Insert 'This is a new line.' at line 1")
            messages.input_text_files = "Line 1\nLine 2"
            
            # Execute with max_turns=1 to allow tool execution
            result = llm(messages=messages, tools=text_editor_tools, max_turns=1)
            
            # Verify the result
            assert result == final_assistant
            assert result.text_files == "Line 1\nThis is a new line.\nLine 2"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_single_turn_text_view(self, mock_backend_setup, text_editor_tools):
        """Test single-turn text viewing with view command."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tool call for text viewing
        tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "view"
            }
        )
        
        # First response with tool call
        first_response = self.create_mock_response(
            "Let me view the text first.",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        
        # Second response after tool execution
        final_response = self.create_mock_response(
            "I can see the text content. It contains a greeting message."
        )
        
        mock_client.chat.completions.create.side_effect = [first_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("Let me view the text first.")
            first_assistant.tool_calls = [tool_call]
            
            # Mock final assistant response
            final_assistant = Assistant("I can see the text content. It contains a greeting message.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input text
            messages = Messages() >> User("What does the text contain?")
            messages.input_text_files = "Hello world! How are you?"
            
            # Execute with max_turns=1 to allow tool execution
            result = llm(messages=messages, tools=text_editor_tools, max_turns=1)
            
            # Verify the result
            assert result == final_assistant
            assert result.text_files == "Hello world! How are you?"  # Unchanged for view
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_multi_turn_text_editing(self, mock_backend_setup, text_editor_tools):
        """Test multi-turn text editing with multiple sequential edits."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # First tool call: replace "Hello" with "Hi"
        first_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "Hello",
                "new_str": "Hi"
            },
            call_id="call_1"
        )
        
        # Second tool call: replace "hi" with "Bonjour"
        second_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "Hi",
                "new_str": "Bonjour"
            },
            call_id="call_2"
        )
        
        # Third tool call: insert exclamation
        third_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "you?",
                "new_str": "you today?!"
            },
            call_id="call_3"
        )
        
        # Mock responses for each turn
        first_response = self.create_mock_response(
            "I'll start by replacing 'Hello' with 'Hi'.",
            tool_calls=[first_tool_call],
            finish_reason="tool_calls"
        )
        
        second_response = self.create_mock_response(
            "Now I'll replace 'Hi' with 'Bonjour'.",
            tool_calls=[second_tool_call],
            finish_reason="tool_calls"
        )
        
        third_response = self.create_mock_response(
            "Finally, let me make it more enthusiastic.",
            tool_calls=[third_tool_call],
            finish_reason="tool_calls"
        )
        
        final_response = self.create_mock_response(
            "I've completed all the text edits as requested!"
        )
        
        mock_client.chat.completions.create.side_effect = [
            first_response, second_response, third_response, final_response
        ]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant responses for each turn
            first_assistant = Assistant("I'll start by replacing 'Hello' with 'Hi'.")
            first_assistant.tool_calls = [first_tool_call]
            
            second_assistant = Assistant("Now I'll replace 'Hi' with 'Bonjour'.")
            second_assistant.tool_calls = [second_tool_call]
            
            third_assistant = Assistant("Finally, let me make it more enthusiastic.")
            third_assistant.tool_calls = [third_tool_call]
            
            final_assistant = Assistant("I've completed all the text edits as requested!")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [
                first_assistant, second_assistant, third_assistant, final_assistant
            ]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input text
            messages = Messages() >> User(
                "Please make these changes to the text: "
                "1. Replace 'Hello' with 'Hi' "
                "2. Replace 'Hi' with 'Bonjour' "
                "3. Make the ending more enthusiastic"
            )
            messages.input_text_files = "Hello world! How are you?"
            
            # Execute with max_turns=3 to allow multiple tool executions
            result = llm(messages=messages, tools=text_editor_tools, max_turns=3)
            
            # Verify the result
            assert result == final_assistant
            assert result.text_files == "Bonjour world! How are you today?!"
            assert mock_client.chat.completions.create.call_count == 4  # 3 tool calls + 1 final
    
    def test_text_editing_error_handling(self, mock_backend_setup, text_editor_tools):
        """Test error handling in text editing when operations fail."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tool call with invalid old_str (not found in text)
        tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "nonexistent",
                "new_str": "replacement"
            }
        )
        
        # First response with tool call
        first_response = self.create_mock_response(
            "I'll try to replace 'nonexistent' text.",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        
        # Second response after tool execution (should handle error)
        final_response = self.create_mock_response(
            "I couldn't find the text 'nonexistent' to replace. The original text remains unchanged."
        )
        
        mock_client.chat.completions.create.side_effect = [first_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("I'll try to replace 'nonexistent' text.")
            first_assistant.tool_calls = [tool_call]
            
            # Mock final assistant response
            final_assistant = Assistant(
                "I couldn't find the text 'nonexistent' to replace. The original text remains unchanged."
            )
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input text
            messages = Messages() >> User("Replace 'nonexistent' with 'replacement'")
            messages.input_text_files = "Hello world! How are you?"
            
            # Execute with max_turns=1 to allow tool execution
            result = llm(messages=messages, tools=text_editor_tools, max_turns=1)
            
            # Verify the result - text should be unchanged due to error
            assert result == final_assistant
            assert result.text_files == "Hello world! How are you?"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_text_editing_with_empty_input(self, mock_backend_setup, text_editor_tools):
        """Test text editing behavior with empty input text."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # Create tool call for insertion in empty text
        tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "insert",
                "insert_line": 0,
                "new_str": "First line of text"
            }
        )
        
        # First response with tool call
        first_response = self.create_mock_response(
            "I'll add the first line to the empty text.",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        
        # Second response after tool execution
        final_response = self.create_mock_response(
            "I've added the first line to your text."
        )
        
        mock_client.chat.completions.create.side_effect = [first_response, final_response]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock first assistant response with tool calls
            first_assistant = Assistant("I'll add the first line to the empty text.")
            first_assistant.tool_calls = [tool_call]
            
            # Mock final assistant response
            final_assistant = Assistant("I've added the first line to your text.")
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [first_assistant, final_assistant]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with empty input text
            messages = Messages() >> User("Add 'First line of text' to the empty document")
            messages.input_text_files = ""
            
            # Execute with max_turns=1 to allow tool execution
            result = llm(messages=messages, tools=text_editor_tools, max_turns=1)
            
            # Verify the result
            assert result == final_assistant
            assert result.text_files == "First line of text\n"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_text_editing_complex_multi_turn_scenario(self, mock_backend_setup, text_editor_tools):
        """Test complex multi-turn scenario with view, replace, and insert operations."""
        mock_backend_class, mock_client = mock_backend_setup
        
        # First tool call: view the text
        view_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {"command": "view"},
            call_id="call_1"
        )
        
        # Second tool call: replace function name
        replace_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "def hello():",
                "new_str": "def greet():"
            },
            call_id="call_2"
        )
        
        # Third tool call: add docstring
        insert_tool_call = self.create_tool_call(
            "str_replace_based_edit_tool",
            {
                "command": "str_replace",
                "old_str": "def greet():",
                "new_str": "def greet():\n    \"\"\"Greet the user.\"\"\""
            },
            call_id="call_3"
        )
        
        # Mock responses for each turn
        view_response = self.create_mock_response(
            "Let me first examine the code.",
            tool_calls=[view_tool_call],
            finish_reason="tool_calls"
        )
        
        replace_response = self.create_mock_response(
            "I'll rename the function from 'hello' to 'greet'.",
            tool_calls=[replace_tool_call],
            finish_reason="tool_calls"
        )
        
        insert_response = self.create_mock_response(
            "Now I'll add a docstring to document the function.",
            tool_calls=[insert_tool_call],
            finish_reason="tool_calls"
        )
        
        final_response = self.create_mock_response(
            "I've successfully refactored the code: renamed the function and added documentation."
        )
        
        mock_client.chat.completions.create.side_effect = [
            view_response, replace_response, insert_response, final_response
        ]
        
        with patch('lmapis.llm.get_backend', return_value=mock_backend_class), \
             patch('lmapis.llm.Assistant.from_model_response') as mock_from_response:
            
            # Mock assistant responses for each turn
            view_assistant = Assistant("Let me first examine the code.")
            view_assistant.tool_calls = [view_tool_call]
            
            replace_assistant = Assistant("I'll rename the function from 'hello' to 'greet'.")
            replace_assistant.tool_calls = [replace_tool_call]
            
            insert_assistant = Assistant("Now I'll add a docstring to document the function.")
            insert_assistant.tool_calls = [insert_tool_call]
            
            final_assistant = Assistant(
                "I've successfully refactored the code: renamed the function and added documentation."
            )
            final_assistant.tool_calls = None
            
            mock_from_response.side_effect = [
                view_assistant, replace_assistant, insert_assistant, final_assistant
            ]
            
            llm = LLM(
                backend="openai",
                model="gpt-4",
                cost={"input": 0.01, "output": 0.03}
            )
            
            # Create messages with input code
            messages = Messages() >> User(
                "Please refactor this code: rename the function to 'greet' and add a docstring"
            )
            messages.input_text_files = 'def hello():\n    print("Hello!")'
            
            # Execute with max_turns=3 to allow multiple tool executions
            result = llm(messages=messages, tools=text_editor_tools, max_turns=3)
            
            # Verify the result
            assert result == final_assistant
            expected_code = 'def greet():\n    """Greet the user."""\n    print("Hello!")'
            assert result.text_files == expected_code
            assert mock_client.chat.completions.create.call_count == 4  # 3 tool calls + 1 final
