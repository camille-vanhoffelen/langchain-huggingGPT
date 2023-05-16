import base64
import json
import logging
import os
import random
from io import BytesIO

import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.prompts import load_prompt

from hugginggpt.exceptions import ModelInferenceException, wrap_exceptions
from hugginggpt.model_factory import TEXT_DAVINCI_003
from hugginggpt.resources import audio_from_bytes, encode_audio, encode_image, get_prompt_resource, get_resource_url, \
    image_from_bytes, load_image, save_audio, save_image
from hugginggpt.task_parsing import Task

logger = logging.getLogger(__name__)
load_dotenv()

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
HUGGINGFACE_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/"


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
    url = HUGGINGFACE_INFERENCE_API_URL + model_id
    huggingface_task = create_huggingface_task(task=task)
    data = huggingface_task.inference_inputs
    response = requests.post(url, headers=HUGGINGFACE_HEADERS, data=data)
    response.raise_for_status()
    result = huggingface_task.parse_response(response)
    logger.debug(f"Inference result: {result}")
    return result


# NLP Tasks
class QuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return {
            "inputs": {
                "question": self.task.args["text"],
                "context": (
                    self.task.args["context"] if "context" in self.task.args else ""
                ),
            }
        }

    def parse_response(self, response):
        return response.json()


# Example added to task-planning-examples.json compared to original paper
class SentenceSimilarity:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        data = {
            "inputs": {
                "source_sentence": self.task.args["text1"],
                "sentences": [self.task.args["text2"]],
            }
        }
        # Using string to bypass requests' form encoding
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


# Example added to task-planning-examples.json compared to original paper
class TextClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]
        # return {"inputs": self.task.args["text"]}

    def parse_response(self, response):
        return response.json()


class TokenClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        return response.json()


# CV Tasks
class VisualQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        img_data = encode_image(self.task.args["image"])
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        data = {
            "inputs": {
                "question": self.task.args["text"],
                "image": img_base64,
            }
        }
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


class DocumentQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        img_data = encode_image(self.task.args["image"])
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        data = {
            "inputs": {
                "question": self.task.args["text"],
                "image": img_base64,
            }
        }
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


class TextToImage:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        image = image_from_bytes(response.content)
        image_name = save_image(image)
        return {"generated image": f"/images/{image_name}.png"}


class ImageSegmentation:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        image_url = get_resource_url(self.task.args["image"])
        image = load_image(image_url)
        colors = []
        for i in range(len(response.json())):
            colors.append(
                (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                    155,
                )
            )
        predicted_results = []
        for i, pred in enumerate(response.json()):
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode="r")
            mask = mask.convert("L")

            layer = Image.new("RGBA", mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
            predicted_results.append(pred)
        image_name = save_image(image)
        return {
            "generated image with segmentation mask": f"/images/{image_name}.png",
            "predicted": predicted_results
        }


class ImageToImage:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        # TODO add optional text argument for control models
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        image = image_from_bytes(response.content)
        image_name = save_image(image)
        return {"generated image": f"/images/{image_name}.png"}


class ObjectDetection:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        image_url = get_resource_url(self.task.args["image"])
        image = load_image(image_url)
        draw = ImageDraw.Draw(image)
        labels = list(item["label"] for item in response.json())
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (
                    random.randint(0, 255),
                    random.randint(0, 100),
                    random.randint(0, 255),
                )
        for item in response.json():
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
            "generated image with predicted box": f"/images/{image_name}.png",
            "predicted": response.json(),
        }


# Example added to task-planning-examples.json compared to original paper
class ImageClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        return response.json()


class ImageToText:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        if "generated_text" in response.json()[0]:
            # TODO why pop here?
            text = response.json()[0].pop("generated_text")
            return {"generated text": text}
        else:
            return {}


# Audio Tasks
class TextToSpeech:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        audio = audio_from_bytes(response.content)
        audio_name = save_audio(audio)
        return {"generated audio": f"/audios/{audio_name}.flac"}


class AudioToAudio:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        result = response.json()
        blob = result[0].items()["blob"]
        content = base64.b64decode(blob.encode("utf-8"))
        audio = audio_from_bytes(content)
        name = save_audio(audio)
        return {"generated audio": f"/audios/{name}.flac"}


class AutomaticSpeechRecognition:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        return response.json()


class AudioClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        return response.json()


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
