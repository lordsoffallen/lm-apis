from lmapis.providers.anthropic import LMApi
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
            {"role": "system", "content": "Answer use queries with simple yes or no"},
            {"role": "user", "content": "Is 2 + 2 = 5?"},
        ]
    )

    print(response)