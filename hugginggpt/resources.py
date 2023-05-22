import os
import uuid
from io import BytesIO

import requests
from PIL import Image
from diffusers.utils.testing_utils import load_image
from pydub import AudioSegment

RESOURCES_DIR = "resources"
PROMPT_TEMPLATES_DIR = "prompt-templates"
GENERATED_RESOURCES_DIR = "output"


def get_prompt_resource(prompt_name: str) -> str:
    return os.path.join(RESOURCES_DIR, PROMPT_TEMPLATES_DIR, prompt_name)


def get_resource_url(resource_arg: str) -> str:
    if resource_arg.startswith("http"):
        return resource_arg
    else:
        return GENERATED_RESOURCES_DIR + resource_arg


# Images
def image_to_bytes(image: Image) -> bytes:
    image_byte = BytesIO()
    image.save(image_byte, format="png")
    image_data = image_byte.getvalue()
    return image_data


def image_from_bytes(img_data: bytes) -> Image:
    return Image.open(BytesIO(img_data))


def encode_image(image_arg: str) -> bytes:
    image_url = get_resource_url(image_arg)
    image = load_image(image_url)
    img_data = image_to_bytes(image)
    return img_data


def save_image(img: Image) -> str:
    name = str(uuid.uuid4())[:4]
    path = f"/images/{name}.png"
    img.save(GENERATED_RESOURCES_DIR + path)
    return path


# Audios
def load_audio(audio_path: str) -> AudioSegment:
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        audio_data = requests.get(audio_path).content
        audio = AudioSegment.from_file(BytesIO(audio_data))
    elif os.path.isfile(audio_path):
        audio = AudioSegment.from_file(audio_path)
    else:
        raise ValueError(
            f"Incorrect path or url, URLs must start with `http://` or `https://`, and {audio_path} is not a valid path"
        )
    return audio


def audio_to_bytes(audio: AudioSegment) -> bytes:
    audio_byte = BytesIO()
    audio.export(audio_byte, format="flac")
    audio_data = audio_byte.getvalue()
    return audio_data


def audio_from_bytes(audio_data: bytes) -> AudioSegment:
    return AudioSegment.from_file(BytesIO(audio_data))


def encode_audio(audio_arg: str) -> bytes:
    audio_url = get_resource_url(audio_arg)
    audio = load_audio(audio_url)
    audio_data = audio_to_bytes(audio)
    return audio_data


def save_audio(audio: AudioSegment) -> str:
    name = str(uuid.uuid4())[:4]
    path = f"/audios/{name}.flac"
    with open(GENERATED_RESOURCES_DIR + path, "wb") as f:
        audio.export(f, format="flac")
    return path


def prepend_resource_dir(s: str) -> str:
    """Prepend the resource dir to all resource paths in the string"""
    for resource_type in ["images", "audios", "videos"]:
        s = s.replace(
            f" /{resource_type}/", f" {GENERATED_RESOURCES_DIR}/{resource_type}/"
        )
    return s


def init_resource_dirs():
    os.makedirs(GENERATED_RESOURCES_DIR + "/images", exist_ok=True)
    os.makedirs(GENERATED_RESOURCES_DIR + "/audios", exist_ok=True)
    os.makedirs(GENERATED_RESOURCES_DIR + "/videos", exist_ok=True)
