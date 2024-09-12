from lmapis.providers.fireworks import LMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_no_args():
    with raises(ValueError):
        LMApi()


@mark.skipif(not is_env_set("FIREWORKS_API_KEY"), reason="Test requires together api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["FIREWORKS_API_KEY"])

    response = llm.client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)
