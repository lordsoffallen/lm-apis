from lmapis.providers.openai import LMApi
from lmapis.utils.messages import Messages, User, Assistant, System, Tool
from pytest import raises, mark
from ..conftest import is_env_set
import warnings


def test_dataclasses():
    content = "Why is sky blue?"
    d = User(content)

    assert d.role == "user"
    assert d.content == content


def test_messages():
    content = "Why is sky blue?"

    output = Messages() >> User(content)

    assert isinstance(output.get(), list)
    assert len(output.get()) == 1
    assert all([isinstance(i, dict) for i in output.get()])


def test_assistant():
    content = "Why is sky blue?"

    output = Messages() >> Assistant(content)

    assert isinstance(output.get(), list)
    assert len(output.get()) == 1
    assert all([isinstance(i, dict) for i in output.get()])

def test_assistant_empty():
    content = "Why is sky blue?"

    output = Messages() >> Assistant(content, tool_calls=[])

    assert isinstance(output.get(), list)
    assert len(output.get()) == 1
    assert all([isinstance(i, dict) for i in output.get()])


class TestTextFileProperties:
    """Test the new text_files properties for User and Assistant messages."""
    
    def test_user_text_files_property(self):
        """Test User text_files property getter and setter."""
        user = User("Hello")
        
        # Test initial state
        assert user.text_files is None
        
        # Test setting single file
        user.text_files = "input.txt"
        assert user.text_files == "input.txt"
        
        # Test setting multiple files
        user.text_files = ["input1.txt", "input2.txt"]
        assert user.text_files == ["input1.txt", "input2.txt"]
    
    def test_assistant_text_files_property(self):
        """Test Assistant text_files property getter and setter."""
        assistant = Assistant("Hello")
        
        # Test initial state
        assert assistant.text_files is None
        
        # Test setting single file
        assistant.text_files = "output.txt"
        assert assistant.text_files == "output.txt"
        
        # Test setting multiple files
        assistant.text_files = ["output1.txt", "output2.txt"]
        assert assistant.text_files == ["output1.txt", "output2.txt"]
    
    def test_tool_call_response_property(self):
        """Test Tool call_response property getter and setter."""
        tool = Tool(content="Tool result", tool_call_id="call_123")
        
        # Test initial state
        assert tool.call_response is None
        
        # Test setting response
        tool.call_response = "Updated content"
        assert tool.call_response == "Updated content"


class TestMessagesFileProperties:
    """Test the new input/output text file properties for Messages class."""
    
    def test_input_text_files_property(self):
        """Test Messages input_text_files property getter and setter."""
        messages = Messages()
        
        # Test initial state
        assert messages.input_text_files is None
        
        # Test setting single file
        messages.input_text_files = "input.txt"
        assert messages.input_text_files == "input.txt"
        
        # Test setting multiple files
        messages.input_text_files = ["input1.txt", "input2.txt"]
        assert messages.input_text_files == ["input1.txt", "input2.txt"]
    
    def test_output_text_files_property(self):
        """Test Messages output_text_files property getter and setter."""
        messages = Messages()
        
        # Test initial state
        assert messages.output_text_files is None
        
        # Test setting single file
        messages.output_text_files = "output.txt"
        assert messages.output_text_files == "output.txt"
        
        # Test setting multiple files
        messages.output_text_files = ["output1.txt", "output2.txt"]
        assert messages.output_text_files == ["output1.txt", "output2.txt"]
    
    def test_messages_auto_file_assignment_user(self):
        """Test that Messages automatically assigns input files from User messages."""
        messages = Messages()
        user = User("Hello")
        user.text_files = "input.txt"
        
        # Add message and check if input files are automatically assigned
        updated_messages = messages >> user
        assert updated_messages.input_text_files == "input.txt"
    
    def test_messages_auto_file_assignment_assistant(self):
        """Test that Messages automatically assigns output files from Assistant messages."""
        messages = Messages()
        assistant = Assistant("Hello")
        assistant.text_files = "output.txt"
        
        # Add message and check if output files are automatically assigned
        updated_messages = messages >> assistant
        assert updated_messages.output_text_files == "output.txt"
    
    def test_messages_no_auto_assignment_when_no_text_files(self):
        """Test that Messages doesn't assign files when message has no text_files."""
        messages = Messages()
        user = User("Hello")
        # Don't set text_files
        
        updated_messages = messages.add_message(user)
        assert updated_messages.input_text_files is None


class TestDeprecationWarnings:
    """Test deprecation warning functionality."""
    
    def test_deprecated_field_warning(self):
        """Test that accessing deprecated fields shows warning."""
        assistant = Assistant("Hello")
        
        # Test that accessing function_call shows deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = assistant.function_call
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)


class TestMessageAsDict:
    """Test the asdict functionality with private attributes."""
    
    def test_base_message_asdict_removes_private_attrs(self):
        """Test that asdict removes attributes starting with __."""
        user = User("Hello")
        user.text_files = "input.txt"  # This sets __file_inputs
        
        result = user.asdict()
        
        # Should not contain private attributes
        assert "__file_inputs" not in result
        assert "role" in result
        assert "content" in result
    
    def test_assistant_asdict_filters_none_values(self):
        """Test that Assistant asdict filters out None and empty list values."""
        assistant = Assistant("Hello")
        assistant.tool_calls = None
        assistant.refusal = None
        
        result = assistant.asdict()
        
        # Should not contain None values
        assert "tool_calls" not in result
        assert "refusal" not in result
        assert "content" in result
        assert result["content"] == "Hello"


@mark.skipif(not is_env_set("OPENAI_API_KEY"),
             reason="Test requires open ai api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["OPENAI_API_KEY"])
    messages = Messages() >> User("Why is sky blue?")

    response = llm.client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )

    print(response)
