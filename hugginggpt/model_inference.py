import base64
import logging
import os
import random
import uuid
from io import BytesIO

import requests
from PIL import Image, ImageDraw
from diffusers.utils.testing_utils import load_image
from huggingface_hub.inference_api import InferenceApi
from langchain import LLMChain, OpenAI
from langchain.prompts import load_prompt
from pydub import AudioSegment

from hugginggpt.exceptions import ModelInferenceException, wrap_exceptions
from hugginggpt.model_factory import TEXT_DAVINCI_003
from hugginggpt.resources import get_prompt_resource
from hugginggpt.task_parsing import Task

logger = logging.getLogger(__name__)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GENERATED_RESOURCES_DIR = "output"


@wrap_exceptions(ModelInferenceException, "Error during model inference")
def infer(task: Task, model_id: str):
    if model_id == "openai":
        return infer_openai(task)
    else:
        return infer_huggingface(task, model_id)


def infer_openai(task: Task):
    logger.info("Starting OpenAI inference")
    prompt_template = load_prompt(
        get_prompt_resource("openai-model-inference-prompt.json")
    )
    # TODO instantiate LLM somewhere else
    llm = OpenAI(model_name=TEXT_DAVINCI_003, temperature=0)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    # Need to replace double quotes with single quotes for correct response generation
    output = llm_chain.predict(
        task=task.json(), task_name=task.task, args=task.args, stop=["<im_end>"]
    )
    result = {"generated text": output}
    logger.debug(f"Inference result: {result}")
    return result


def infer_huggingface(task: Task, model_id: str):
    logger.info("Starting huggingface inference")
    # TODO validate inference output
    inference = InferenceApi(repo_id=model_id, token=HUGGINGFACE_TOKEN)
    huggingface_task = create_huggingface_task(task=task)
    response = inference(
        inputs=huggingface_task.inference_inputs,
        raw_response=huggingface_task.raw_response,
    )
    logger.debug(f"Huggingface inference response: {response}")
    check_for_errors(response)
    result = huggingface_task.parse_response(response)
    logger.debug(f"Inference result: {result}")
    return result


# NLP Tasks
class QuestionAnswering:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {
            "question": self.task.args["text"],
            "context": (
                self.task.args["context"] if "context" in self.task.args else ""
            ),
        }

    def parse_response(self, response):
        return response


class SentenceSimilarity:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {
            "source_sentence": self.task.args["text1"],
            "target_sentence": self.task.args["text2"],
        }

    def parse_response(self, response):
        return response


class TextClassification:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        return response


class TokenClassification:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        return response


# CV Tasks
class VisualQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {
            "question": self.task.args["text"],
            "image": encode_image(self.task.args["image"]),
        }

    def parse_response(self, response):
        return response


class DocumentQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {
            "question": self.task.args["text"],
            "image": encode_image(self.task.args["image"]),
        }

    def parse_response(self, response):
        return response


class TextToImage:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        image_name = save_image(response)
        return {"generated image": f"/images/{image_name}.png"}


class ImageSegmentation:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"image": encode_image(self.task.args["image"])}

    def parse_response(self, response):
        img_url = get_resource_url(self.task.args["image"])
        img_data = image_to_bytes(img_url)
        image = Image.open(BytesIO(img_data))
        colors = []
        for i in range(len(response)):
            colors.append(
                (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                    155,
                )
            )
        for i, pred in enumerate(response):
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode="r")
            mask = mask.convert("L")

            layer = Image.new("RGBA", mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
        image_name = save_image(image)
        return {
            "generated image with segmentation mask": f"/images/{image_name}.png",
            "predicted": response,
        }


class ImageToImage:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"image": encode_image(self.task.args["image"])}

    def parse_response(self, response):
        image_name = save_image(response)
        return {"generated image": f"/images/{image_name}.png"}


class ObjectDetection:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"image": encode_image(self.task.args["image"])}

    def parse_response(self, response):
        img_url = get_resource_url(self.task.args["image"])
        img_data = image_to_bytes(img_url)
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item["label"] for item in response)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (
                    random.randint(0, 255),
                    random.randint(0, 100),
                    random.randint(0, 255),
                )
        for item in response:
            box = item["box"]
            draw.rectangle(
                ((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])),
                outline=color_map[item["label"]],
                width=2,
            )
            draw.text(
                (box["xmin"] + 5, box["ymin"] - 15),
                item["label"],
                fill=color_map[item["label"]],
            )
        image_name = save_image(image)
        return {
            "generated image with predicted box": f"/images/{image_name}.jpg",
            "predicted": response,
        }


class ImageClassification:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"image": encode_image(self.task.args["image"])}

    def parse_response(self, response):
        return response


class ImageToText:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"image": encode_image(self.task.args["image"])}

    def parse_response(self, response):
        if "generated_text" in response[0]:
            # TODO why pop here?
            text = response[0].pop("generated_text")
            return {"generated text": text}
        else:
            return {}


# Audio Tasks
class TextToSpeech:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = True

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        audio_name = save_audio(response.content)
        return {"generated audio": f"/audios/{audio_name}.flac"}


class AudioToAudio:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = True

    @property
    def inference_inputs(self):
        return {"audio": encode_audio(self.task.args["audio"])}

    def parse_response(self, response):
        result = response.json()
        blob = result[0].items()["blob"]
        content = base64.b64decode(blob.encode("utf-8"))
        # TODO can this just be saved like for text to speech?
        audio = AudioSegment.from_file(BytesIO(content))
        name = str(uuid.uuid4())[:4]
        audio.export(f"{GENERATED_RESOURCES_DIR}/audios/{name}.flac", format="flac")
        return {"generated audio": f"/audios/{name}.flac"}


class AutomaticSpeechRecognition:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"audio": encode_audio(self.task.args["audio"])}

    def parse_response(self, response):
        return response


class AudioClassification:
    def __init__(self, task: Task):
        self.task = task
        self.raw_response = False

    @property
    def inference_inputs(self):
        return {"audio": encode_audio(self.task.args["audio"])}

    def parse_response(self, response):
        return response


HUGGINGFACE_TASKS = {
    "question-answering": QuestionAnswering,
    "sentence-similarity": SentenceSimilarity,
    "text-classification": TextClassification,
    "token-classification": TokenClassification,
    "visual-question-answering": VisualQuestionAnswering,
    "document-question-answering": DocumentQuestionAnswering,
    "text-to-image": TextToImage,
    "image-segmentation": ImageSegmentation,
    "image-to-image": ImageToImage,
    "object-detection": ObjectDetection,
    "image-classification": ImageClassification,
    "image-to-text": ImageToText,
    "text-to-speech": TextToSpeech,
    "automatic-speech-recognition": AutomaticSpeechRecognition,
    "audio-to-audio": AudioToAudio,
    "audio-classification": AudioClassification,
}


def create_huggingface_task(task: Task):
    if task.task in HUGGINGFACE_TASKS:
        return HUGGINGFACE_TASKS[task.task](task)
    else:
        raise NotImplementedError(f"Task {task.task} not supported")


def get_resource_url(resource_arg):
    if resource_arg.startswith("http"):
        return resource_arg
    else:
        return GENERATED_RESOURCES_DIR + resource_arg


def image_to_bytes(img_url):
    img_byte = BytesIO()
    type = img_url.split(".")[-1]
    load_image(img_url).save(img_byte, format="png")
    img_data = img_byte.getvalue()
    return img_data


def encode_image(image_arg):
    img_url = get_resource_url(image_arg)
    img_data = image_to_bytes(img_url)
    return base64.b64encode(img_data).decode("utf-8")


def encode_audio(audio_arg):
    audio_url = get_resource_url(audio_arg)
    audio_data = requests.get(audio_url, timeout=10).content
    return base64.b64encode(audio_data).decode("utf-8")


def save_image(img: Image):
    name = str(uuid.uuid4())[:4]
    path = f"{GENERATED_RESOURCES_DIR}/images/{name}.png"
    img.save(path)
    return name


def save_audio(audio):
    name = str(uuid.uuid4())[:4]
    with open(f"{GENERATED_RESOURCES_DIR}/audios/{name}.flac", "wb") as f:
        f.write(audio)
    return name


def check_for_errors(response):
    if isinstance(response, dict) and "error" in response:
        raise Exception(f"Error response: {response['error']}")