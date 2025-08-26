import json

from anthropic import NOT_GIVEN
from lmapis.providers.anthropic import LMApi, AsyncLMApi, _convert_single_message, \
    convert_messages, convert_tools
from lmapis.utils.messages import Assistant
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_no_args():
    with raises(ValueError):
        LMApi()


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_api(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_api_response_parsing(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    # Test response parsing
    assistant = Assistant.from_model_response(response)
    print(assistant)


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_api_image_response_error(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    with raises(NotImplementedError):
        llm.client.chat.completions.create(
            model="claude-3-haiku-20240307",
            messages=[{
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": {
                        "url": "test-image-url.com",
                        "detail": "auto"
                    }
                }
            }]
        )


@mark.asyncio
@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
async def test_api_async(envs):
    llm = AsyncLMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = await llm.client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)


def test_non_supported_args():
    llm = LMApi("test")

    with raises(ValueError):
        llm.client.chat.completions.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
            logprobs="0.5"  # not supported in claude
        )


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_api_multi_system_prompt(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[
            {"role": "system", "content": "You are an expert AI assistant at math"},
            {"role": "system", "content": "Answer user queries with simple yes or no"},
            {"role": "user", "content": "Is 2 + 2 = 5?"},
        ]
    )

    print(response)


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_api_max_tokens(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[
            {"role": "system", "content": "You are an expert AI assistant at math"},
            {"role": "system", "content": "Answer user queries with a detailed explanation"},
            {"role": "user", "content": "Is 2 + 2 = 5?"},
        ],
        max_tokens=10
    )

    assert response.choices[0].finish_reason == "length"

    print(response)


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_tool_calls(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {
                "role": "user",
                "content": "There's a syntax error in my primes.py file. Can you help me fix it?"
            }
        ],
        max_tokens=500,
        tools=[
            {
                "type": "text_editor_20250124",
                "name": "str_replace_editor"
            }
        ],
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert len(response.choices[0].message.tool_calls) > 0
    assert "str_replace_editor" == response.choices[0].message.tool_calls[0].function.name

    print(response)


@mark.skipif(
    not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires anthropic api key to run"
)
def test_response_asdict(envs):
    llm = LMApi(api_key=envs["ANTHROPIC_API_KEY"])

    response = llm.client.chat.completions.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {
                "role": "user",
                "content": "There's a syntax error in my primes.py file. Can you help me fix it?"
            }
        ],
        max_tokens=500,
        tools=[
            {
                "type": "text_editor_20250124",
                "name": "str_replace_editor"
            }
        ],
    )

    assistant = Assistant.from_model_response(response)
    d = assistant.asdict()
    # Check if we can dump as json
    j = json.dumps(d)


def test_convert_single_message_user():
    msg = {"role": "user", "content": "Hello, world!"}
    result = _convert_single_message(msg)
    
    assert result["role"] == "user"
    assert result["content"] == "Hello, world!"


def test_convert_single_message_system():
    msg = {"role": "system", "content": "You are a helpful assistant"}
    result = _convert_single_message(msg)
    
    assert result["role"] == "system"
    assert result["content"] == "You are a helpful assistant"


def test_convert_single_message_assistant():
    msg = {"role": "assistant", "content": "I'm here to help!"}
    result = _convert_single_message(msg)
    
    assert result["role"] == "assistant"
    assert result["content"] == "I'm here to help!"


def test_convert_single_message_assistant_no_tool():
    msg = {"role": "assistant", "content": "I'm here to help!", 'tool_calls': None}
    result = _convert_single_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "I'm here to help!"


def test_convert_single_message_tool_response():
    msg = {
        "role": "tool",
        "content": "Function executed successfully",
        "tool_call_id": "call_123"
    }
    result = _convert_single_message(msg)
    
    assert result["role"] == "user"
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "tool_result"
    assert result["content"][0]["tool_use_id"] == "call_123"
    assert result["content"][0]["content"] == "Function executed successfully"
    assert result["content"][0]["is_error"] == False


def test_convert_single_message_tool_response_with_error():
    msg = {
        "role": "tool",
        "content": "Error: Function failed to execute",
        "tool_call_id": "call_456"
    }
    result = _convert_single_message(msg)
    
    assert result["role"] == "user"
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "tool_result"
    assert result["content"][0]["tool_use_id"] == "call_456"
    assert result["content"][0]["content"] == "Error: Function failed to execute"
    assert result["content"][0]["is_error"] == True


def test_convert_single_message_assistant_with_tool_calls():
    msg = {
        "role": "assistant",
        "content": "I'll help you with that calculation.",
        "tool_calls": [
            {
                "id": "call_789",
                "function": {
                    "name": "calculate",
                    "arguments": '{"operation": "add", "a": 5, "b": 3}'
                }
            }
        ]
    }
    result = _convert_single_message(msg)
    
    assert result["role"] == "assistant"
    assert len(result["content"]) == 2
    
    # Check text content
    text_block = result["content"][0]
    assert text_block.type == "text"
    assert text_block.text == "I'll help you with that calculation."
    
    # Check tool use block
    tool_block = result["content"][1]
    assert tool_block["type"] == "tool_use"
    assert tool_block["id"] == "call_789"
    assert tool_block["name"] == "calculate"
    assert tool_block["input"] == {"operation": "add", "a": 5, "b": 3}


def test_convert_single_message_assistant_tool_calls_only():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_999",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "New York"}'
                }
            }
        ]
    }
    result = _convert_single_message(msg)
    
    assert result["role"] == "assistant"
    assert len(result["content"]) == 1
    
    # Check tool use block
    tool_block = result["content"][0]
    assert tool_block["type"] == "tool_use"
    assert tool_block["id"] == "call_999"
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"location": "New York"}


def test_convert_single_message_image_not_implemented():
    msg = {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    }
    
    with raises(NotImplementedError, match="Image parsing is not implemented yet"):
        _convert_single_message(msg)


def test_convert_messages_with_system():
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    system, converted = convert_messages(messages)
    
    # Check system prompts were extracted
    assert len(system) == 2
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "You are helpful"
    assert system[1]["type"] == "text"
    assert system[1]["text"] == "Be concise"
    
    # Check remaining messages
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "Hello"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == "Hi there!"


def test_convert_messages_no_system():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    system, converted = convert_messages(messages)
    
    # Check no system prompts
    assert system == NOT_GIVEN
    
    # Check messages
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"


def test_convert_tools_openai_format():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    result = convert_tools(tools)
    
    assert len(result) == 1
    tool = result[0]
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Get current weather"
    assert tool["input_schema"]["type"] == "object"
    assert tool["input_schema"]["properties"]["location"]["type"] == "string"
    assert tool["input_schema"]["required"] == ["location"]


def test_convert_tools_anthropic_native():
    tools = [
        {
            "type": "text_editor_20250124",
            "name": "str_replace_editor"
        }
    ]
    
    result = convert_tools(tools)
    
    assert len(result) == 1
    assert result[0] == tools[0]  # Should pass through unchanged


def test_convert_tools_mixed():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "text_editor_20250124",
            "name": "str_replace_editor"
        }
    ]
    
    result = convert_tools(tools)
    
    assert len(result) == 2
    
    # First tool should be converted
    assert result[0]["name"] == "calculate"
    assert result[0]["description"] == "Perform calculation"
    
    # Second tool should pass through
    assert result[1]["type"] == "text_editor_20250124"
    assert result[1]["name"] == "str_replace_editor"


def test_convert_tools_none():
    """Test converting None tools"""
    from lmapis.providers.anthropic import convert_tools
    from anthropic import NOT_GIVEN
    
    result = convert_tools(None)
    assert result == NOT_GIVEN
    
    result = convert_tools(NOT_GIVEN)
    assert result == NOT_GIVEN


def test_convert_tools_no_required_params():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "random_number",
                "description": "Generate random number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "integer"},
                        "max": {"type": "integer"}
                    }
                }
            }
        }
    ]
    
    result = convert_tools(tools)
    
    assert len(result) == 1
    tool = result[0]
    assert tool["name"] == "random_number"
    assert tool["input_schema"]["required"] == []  # Should default to empty list 