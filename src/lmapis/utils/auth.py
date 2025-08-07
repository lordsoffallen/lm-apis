import os


def get_api_key_from_env(name: str) -> str:
    """Get API key from environment variables."""
    try:
       return os.environ[name]
    except KeyError:
        raise ValueError(
            f"{name} required. Either set {name} as env variable or pass the api_key"
        )