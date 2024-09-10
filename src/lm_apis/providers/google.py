from lm_apis.base import BaseLMApi
from lm_apis.logging import get_logger

try:
    from google.auth import default
    from google.auth.transport.requests import Request
except ImportError:
    raise ImportError(
        "Using Google Backend requires "
        "`google-auth`,`requests` lib to be install"
    )


logger = get_logger(__file__)

SUPPORTED_REGIONS = frozenset(
    {
        "africa-south1",
        "asia-east1",
        "asia-east2",
        "asia-northeast1",
        "asia-northeast2",
        "asia-northeast3",
        "asia-south1",
        "asia-southeast1",
        "asia-southeast2",
        "australia-southeast1",
        "australia-southeast2",
        "europe-central2",
        "europe-north1",
        "europe-southwest1",
        "europe-west1",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west8",
        "europe-west9",
        "europe-west12",
        "me-central1",
        "me-central2",
        "me-west1",
        "northamerica-northeast1",
        "northamerica-northeast2",
        "southamerica-east1",
        "southamerica-west1",
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west2",
        "us-west3",
        "us-west4",
    }
)


def validate_region(region: str):
    """Validates region against supported regions.

    Args:
        region: region to validate

    Raises:
        ValueError: If region is not in supported regions.
    """
    if not region:
        raise ValueError(
            f"Please provide a region, select from {SUPPORTED_REGIONS}"
        )

    region = region.lower()
    if region not in SUPPORTED_REGIONS:
        raise ValueError(
            f"Unsupported region for Vertex AI, select from {SUPPORTED_REGIONS}"
        )


class LMApi(BaseLMApi):
    def __init__(
        self,
        region: str,
        project_id: str = None,
        endpoint: str = "openapi",
        api_key: str = None,
        auto_refresh_api_key: bool = True,
    ):
        """
        Implementation for Google models wrapper around OpenAI endpoint. The authentication
        is described in `google.auth.default` function. For further infor, take a look at there.

        Args:
            region: Google region
            project_id: Google project id. If not provided, will attempt to infer from an env
            endpoint: Define if you would like to use model garden model in which case
                the model id should be passed, otherwise defaults to openapi
            api_key: Authentication variable, if none it will try to use `get_access_token`
                function to programmatically retrieve access token from Google
            auto_refresh_api_key: If set to True, credentials  refreshed automatically
                during runtime without calls being failed. Does not have any effect when
                api_key is passed
        """

        if endpoint is None:
            raise ValueError("endpoint can not be None")

        self.project_id = project_id
        self.auto_refresh_api_key = auto_refresh_api_key

        if api_key is None:
            # Programmatically get an access token\
            # Note: the credential lives for 1 hour by default
            # (https://cloud.google.com/docs/authentication/token-types#at-lifetime);
            # after expiration, it must be refreshed.
            self._creds, project = default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
                quota_project_id=project_id,
            )

            if project_id is None:
                # If user didn't provide but default config is mapped
                self.project_id = project

            if self.project_id is None:
                raise ValueError(
                    "Project ID could not be determined from the default settings. "
                    "Please provide this or set the right environment variables"
                )

            # Set a dummy api key as we override during runtime
            self.api_key = "i-will-be-overwritten-during-runtime"
        else:
            self.auto_refresh_api_key = False   # User passed an api key, not needed
            self.api_key = api_key

            if project_id is None:
                logger.error(
                    "Project ID is not passed, but api_key was passed. Project ID is "
                    "required when api_key is passed"
                )

        validate_region(region)

        self.url = f'https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/{endpoint}'

        super().__init__(base_url=self.url, api_key=self.api_key)

    def _refresh_creds_if_expired(self):
        # Automatically refresh creds from Google if expired
        if not self._creds.valid:
            auth_req = Request()
            self._creds.refresh(auth_req)

            if not self._creds.valid:
                raise RuntimeError("Unable to refresh auth")

            self._client.api_key = self._creds.token

    @property
    def client(self):
        client = super().client
        if self.auto_refresh_api_key:
            self._refresh_creds_if_expired()

        return client
