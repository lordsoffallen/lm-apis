import os

from lmapis.base import BaseLMApi, BaseAsyncLMApi
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
from anthropic.resources.messages import cached_property

# Import the local files
from .anthropic import ChatAnthropic, AsyncChatAnthropic


class LMApi(BaseLMApi):
    def __init__(self, aws_region:str, api_key: str = None, **kwargs):
        """
        Authenticate by either providing the keys below or use the default AWS
        credential providers, such as using ~/.aws/credentials or the
        "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.

        aws_access_key="<access key>",
        aws_secret_key="<secret key>",

        Temporary credentials can be used with aws_session_token.
        Read more at https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html.
        aws_session_token="<session_token>",
        """
        self.aws_region = aws_region or os.getenv("AWS_REGION")

        if self.aws_region is None:
            raise ValueError("Make sure AWS region is defined properly for bedrock")

        super().__init__(
            base_url=None, api_key=api_key, **kwargs
        )

    @property
    def client(self):
        if self._client is None:
            self._client = LMApiAnthropic(aws_region=self.aws_region, **self.kwargs)
        return self._client


class LMApiAnthropic(AnthropicBedrock):
    @cached_property
    def chat(self):
        return ChatAnthropic(self)


class AsyncLMApi(BaseAsyncLMApi):
    def __init__(self, aws_region:str, api_key: str = None, **kwargs):
        """
        Authenticate by either providing the keys below or use the default AWS
        credential providers, such as using ~/.aws/credentials or the
        "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.

        aws_access_key="<access key>",
        aws_secret_key="<secret key>",

        Temporary credentials can be used with aws_session_token.
        Read more at https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html.
        aws_session_token="<session_token>",
        """
        self.aws_region = aws_region or os.getenv("AWS_REGION")

        if self.aws_region is None:
            raise ValueError("Make sure AWS region is defined properly for bedrock")

        super().__init__(
            base_url=None, api_key=api_key, **kwargs
        )

    @property
    def client(self):
        if self._client is None:
            self._client = AsyncLMApiAnthropic(aws_region=self.aws_region, **self.kwargs)
        return self._client


class AsyncLMApiAnthropic(AsyncAnthropicBedrock):
    @cached_property
    def chat(self):
        return AsyncChatAnthropic(self)
