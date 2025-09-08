from lmapis.providers.anthropic_bedrock import LMApi, AsyncLMApi
from ..conftest import is_env_set
from pytest import mark


def is_aws_envs_set():
    return is_env_set("AWS_REGION") and \
        is_env_set("AWS_SECRET_ACCESS_KEY") and \
        is_env_set("AWS_ACCESS_KEY_ID")


@mark.skipif(not is_aws_envs_set(), reason="Test requires AWS envs to run")
def test_no_api_key(envs):
    llm = LMApi(
        aws_secret_key=envs["AWS_SECRET_ACCESS_KEY"],
        aws_access_key=envs["AWS_ACCESS_KEY_ID"],
        aws_region=envs["AWS_REGION"]
    )

    response = llm.client.chat.completions.create(
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)


@mark.asyncio
@mark.skipif(
    not is_aws_envs_set(), reason="Test requires project id to run"
)
async def test_api_async(envs):
    llm = AsyncLMApi(
        aws_secret_key=envs["AWS_SECRET_ACCESS_KEY"],
        aws_access_key=envs["AWS_ACCESS_KEY_ID"],
        aws_region=envs["AWS_REGION"]
    )

    response = await llm.client.chat.completions.create(
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)
