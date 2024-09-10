from dotenv import dotenv_values
from pytest import fixture


def _get_env_values():
    return dotenv_values(verbose=True)


@fixture(scope="session")
def envs():
    return _get_env_values()


def is_env_set(name):
    env = _get_env_values()
    if env.get(name) is not None:
        return True
    return False