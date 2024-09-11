from lmapis.providers.openai import LMApi
from lmapis.utils import Messages, User
from pytest import raises, mark
from .conftest import is_env_set


def test_dataclasses():
    content = "Why is sky blue?"
    d = User(content)

    assert d.role == "user"
    assert d.content == content


def test_messages():
    content = "Why is sky blue?"
    d = User(content)

    output = Messages(d, d)

    assert isinstance(output, list)
    assert len(output) == 2
    assert all([isinstance(i, dict) for i in output])


@mark.skipif(not is_env_set("OPENAI_API_KEY"),
             reason="Test requires open ai api key to run")
def test_api(envs):
    llm = LMApi(api_key=envs["OPENAI_API_KEY"])

    response = llm.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=Messages(
            User("Why is sky blue?")
        )
    )

    print(response)
