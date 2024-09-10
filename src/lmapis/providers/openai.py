from lmapis.base import BaseLMApi, BaseAsyncLMApi
from lmapis.utils import get_api_key_from_env


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("OPENAI_API_KEY")

        super().__init__(api_key=api_key, **kwargs)


class AsyncLMApi(BaseAsyncLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("OPENAI_API_KEY")

        super().__init__(api_key=api_key, **kwargs)
