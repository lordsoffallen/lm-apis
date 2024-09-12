from lmapis.base import BaseLMApi
from lmapis.utils import get_api_key_from_env


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("TOGETHER_API_KEY")

        super().__init__(
            base_url="https://api.together.xyz/v1", api_key=api_key, **kwargs
        )
