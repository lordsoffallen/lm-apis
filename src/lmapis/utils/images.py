from typing import Literal
from dataclasses import dataclass, field
from io import BytesIO
import base64


def get_image_bytes(image: "Image.Image", file_format="PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffered = BytesIO()
    image.save(buffered, format=file_format)
    img_bytes = buffered.getvalue()
    return img_bytes


def encode_image(image: "Image.Image", file_format="PNG") -> str:
    """Encode PIL Image to base64 data URL."""
    img_bytes = get_image_bytes(image, file_format)
    encoded_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{file_format};base64,{encoded_str}"


@dataclass
class ImageURL:
    url: str  # Either a URL of the image or the base64 encoded image data.
    detail: Literal["auto", "low", "high"]


@dataclass
class ImageContent:
    image_url: ImageURL
    type: str = field(default="image_url", init=False)