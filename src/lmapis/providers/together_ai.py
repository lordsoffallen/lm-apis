from lmapis.base import BaseLMApi
from lmapis.logging import get_logger

import os

logger = get_logger(__file__)


class LMApi(BaseLMApi):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            try:
                api_key = os.environ["TOGETHER_API_KEY"]
            except KeyError as e:
                logger.error(
                    "Together API key required. Set TOGETHER_API_KEY or "
                    "pass api_key variable"
                )
                raise e

        super().__init__(
            base_url="https://api.together.xyz/v1", api_key=api_key, **kwargs
        )
