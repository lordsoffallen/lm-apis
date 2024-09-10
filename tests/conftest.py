from dotenv import dotenv_values
from pathlib import Path
from pytest import fixture


PROJECT_PATH = Path(__file__).parent.parent.resolve()   # Main lm-apis dir


def _get_env_values():
    return dotenv_values(PROJECT_PATH, verbose=True)

@fixture(scope="session")
def envs():
    return _get_env_values()


def is_env_set(name):
    env = _get_env_values()
    if env.get(name) is not None:
        return True
    return False