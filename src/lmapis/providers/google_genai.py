from lmapis.base import BaseLMApi, BaseAsyncLMApi
from lmapis.utils.auth import get_api_key_from_env


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = get_api_key_from_env("GEMINI_API_KEY")

        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )


class AsyncLMApi(BaseAsyncLMApi):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = get_api_key_from_env("GEMINI_API_KEY")

        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
