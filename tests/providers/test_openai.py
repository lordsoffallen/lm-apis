from lmapis.providers.openai import LMApi, AsyncLMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_error():
    with raises(ValueError):
        LMApi()


@mark.skipif(not is_env_set("OPENAI_API_KEY"),
             reason="Test requires open ai api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["OPENAI_API_KEY"])

    response = llm.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)


@mark.asyncio
@mark.skipif(not is_env_set("OPENAI_API_KEY"),
             reason="Test requires open ai api key to run")
async def test_api_async(envs):
    llm = AsyncLMApi(api_key=envs["OPENAI_API_KEY"])

    response = await llm.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)
