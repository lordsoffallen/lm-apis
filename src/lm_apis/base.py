from openai import OpenAI, AsyncOpenAI


class BaseLMApi:
    def __init__(self, base_url: str, api_key: str = None, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.kwargs = kwargs
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, **self.kwargs
            )
        return self._client


class BaseAsyncLMApi(BaseLMApi):
    @property
    def client(self):
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, **self.kwargs
            )
        return self._client
