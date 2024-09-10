from lmapis.providers.google import LMApi
from ..conftest import is_env_set
from pytest import raises, mark
from os import environ


def test_region_only():
    # no project id passed not env variables are set
    with raises(ValueError):
        LMApi(region="europe-west1")

    with raises(ValueError):    # Test wrong region name
        LMApi(region="eurasdas-west1")


def _is_gcp_project_available() -> bool:
    return (is_env_set("GOOGLE_PROJECT_ID") or
            (environ.get("GOOGLE_CLOUD_PROJECT") is not None))


def _is_gcp_api_key_available() -> bool:
    return is_env_set("GOOGLE_API_KEY")


@mark.skipif(not _is_gcp_project_available(), reason="Test requires project id to run")
def test_no_api_key(envs):

    llm = LMApi(region="europe-west1", project_id=envs["GOOGLE_PROJECT_ID"])

    response = llm.client.chat.completions.create(
        model="google/gemini-1.5-flash-001",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)


@mark.skipif(
    not (_is_gcp_project_available() and _is_gcp_api_key_available()),
    reason="Test requires project id to run"
)
def test_api_key(envs):
    llm = LMApi(
        region="europe-west1",
        project_id=envs["GOOGLE_PROJECT_ID"],
        api_key=envs["GOOGLE_API_KEY"]
    )

    response = llm.client.chat.completions.create(
        model="google/gemini-1.5-flash-001",
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )

    print(response)