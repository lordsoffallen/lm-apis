from lmapis.base import BaseLMApi, BaseAsyncLMApi
from lmapis.utils import get_api_key_from_env
from mistralai import Mistral, Chat, SDKError

import time


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
            self._client.chat.completions = CompletionsMistral(self._client.chat)
        return self._client


class CompletionsMistral(Chat):
    def __init__(self, chat_client: Chat):
        self.chat_client = chat_client

    def create(self, *args, **kwargs):
        try:
            return self.chat_client.complete(*args, **kwargs)
        except SDKError as e:
            if e.status_code == 429:
                time.sleep(1.1)     # sleep 1 second, mistral does 1rps
                return self.chat_client.complete(*args, **kwargs)
            raise e


class AsyncLMApi(BaseAsyncLMApi):
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
            self._client.chat.completions = AsyncCompletionsMistral(self._client.chat)
        return self._client


class AsyncCompletionsMistral(CompletionsMistral):
    def create(self, *args, **kwargs):
        try:
            return self.chat_client.complete_async(*args, **kwargs)
        except SDKError as e:
            if e.status_code == 429:
                time.sleep(1.1)     # sleep 1 second, mistral does 1rps
                return self.chat_client.complete_async(*args, **kwargs)
            raise e
