from lmapis.base import BaseLMApi
from lmapis.logging import get_logger
from lmapis.utils import get_api_key_from_env


logger = get_logger(__file__)


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = get_api_key_from_env("GEMINI_API_KEY")

        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
