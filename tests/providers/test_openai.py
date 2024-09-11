from lmapis.providers.openai import LMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api():
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
