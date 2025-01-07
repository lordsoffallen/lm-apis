from lmapis.base import BaseLMApi
from lmapis.utils import get_api_key_from_env
from mistralai import Mistral


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = get_api_key_from_env("MISTRAL_API_KEY")

        super().__init__(
            base_url="https://api.mistral.ai/v1", api_key=api_key, **kwargs
        )

    @property
    def client(self):
        if self._client is None:
            self._client = Mistral(
                # server_url=self.base_url,
                api_key=self.api_key,
                **self.kwargs
            )
        return self._client
