from lmapis.providers.together import LMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_no_args():
    with raises(ValueError):
        LMApi()


@mark.skipif(not is_env_set("TOGETHER_API_KEY"), reason="Test requires together api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["TOGETHER_API_KEY"])

    response = llm.client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)
