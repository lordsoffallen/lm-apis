from lmapis.providers.google_genai import LMApi, AsyncLMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_no_args():
    with raises(ValueError):
        LMApi()


@mark.skipif(not is_env_set("GEMINI_API_KEY"), reason="Test requires gemini api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["GEMINI_API_KEY"])

    response = llm.client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "Why is the sky blue? Explain briefly"}]
    )

    print(response)


@mark.asyncio
@mark.skipif(not is_env_set("GEMINI_API_KEY"), reason="Test requires gemini api key to run")
async def test_api_async(envs):
    llm = AsyncLMApi(api_key=envs["GEMINI_API_KEY"])

    response = await llm.client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "Why is the sky blue? Explain briefly"}]
    )

    print(response)
