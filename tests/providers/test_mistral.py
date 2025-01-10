from lmapis.providers.mistral import LMApi
from pytest import raises, mark
from ..conftest import is_env_set


def test_api_no_args():
    with raises(ValueError):
        LMApi()


@mark.skipif(
    not is_env_set("MISTRAL_API_KEY"), reason="Test requires mistral api key to run"
)
def test_api(envs):
    llm = LMApi(api_key=envs["MISTRAL_API_KEY"])

    response = llm.client.chat.completions.create(
        model="open-mistral-nemo",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)
