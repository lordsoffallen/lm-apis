from typing import Literal
from dataclasses import dataclass, field, asdict

import os



def get_api_key_from_env(name: str) -> str:
    try:
       return os.environ[name]
    except KeyError:
        raise ValueError(
            f"{name} required. Either set {name} as env variable or pass the api_key"
        )


@dataclass
class ImageURL:
    url: str  # Either a URL of the image or the base64 encoded image data.
    detail: Literal["auto", "low", "high"]


@dataclass
class ImageContent:
    image_url: ImageURL
    type: str


@dataclass
class TextContent:
    text: str
    type = "text"


@dataclass
class BaseMessage:
    role: str
    content: str    # TODO add image support later TextContent | ImageContent


def Messages(*args) -> list[dict]:  # noqa
    """ Use this method to encapsulate API call endpoint for dataclasses"""
    return [asdict(arg) for arg in args]


@dataclass
class System(BaseMessage):
    role: str = field(default="system", init=False)


@dataclass
class User(BaseMessage):
    role: str = field(default="user", init=False)


@dataclass
class Assistant(BaseMessage):       # TODO extend support later
    role: str = field(default="assistant", init=False)


@dataclass
class Tool(BaseMessage):
    role: str = field(default="tool", init=False)
    tool_call_id: str


@dataclass
class Function(BaseMessage):
    role: str = field(default="function", init=False)
    name: str
