from lmapis.providers.anthropic import LMApi, AsyncLMApi
from lmapis.utils import Assistant
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