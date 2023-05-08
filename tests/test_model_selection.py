import json

import pytest
from dotenv import load_dotenv
from langchain.llms.fake import FakeListLLM
from langchain.prompts import load_prompt

from hugginggpt.model_selection import Model, select_model
from hugginggpt.task_parsing import Task
from hugginggpt.resources import get_prompt_resource

load_dotenv()


def test_model_selection_prompt(model_selection_prompt, user_input, task, models):
    prompt_template = load_prompt(get_prompt_resource("model-selection-prompt.json"))
    prompt = prompt_template.format(
        user_input=user_input, task=task.json(), models=json.dumps(models)
    )
    assert prompt == model_selection_prompt


def test_select_model(user_input, task, models, llm, expected_model):
    model = select_model(user_input, task, models, llm)
    assert model == expected_model


@pytest.fixture
def id():
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def reason():
    return "This model has the most potential to solve the user request as it has the highest number of likes and the most detailed description"


@pytest.fixture
def response(id, reason):
    return f'{{"id": "{id}", "reason": "{reason}"}}'


@pytest.fixture
def llm(response):
    llm = FakeListLLM(responses=[response])
    return llm


@pytest.fixture
def expected_model(id, reason):
    return Model(id=id, reason=reason)


@pytest.fixture
def model_selection_prompt():
    with open("tests/resources/model-selection-completion.txt", "r") as f:
        return f.read()


@pytest.fixture
def user_input():
    return "Draw me a sheep."


@pytest.fixture
def task():
    return Task(task="text-to-image", id=0, dep=[-1], args={"text": "Draw me a sheep."})


@pytest.fixture
def models():
    return [
        {
            "id": "runwayml/stable-diffusion-v1-5",
            "inference endpoint": [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2",
                "WarriorMama777/OrangeMixs",
                "andite/anything-v4.0",
                "prompthero/openjourney",
            ],
            "likes": 6367,
            "description": "\n\n# Stable Diffusion v1-5 Model Card\n\nStable Diffusion is a latent text-to-image diffusion model cap",
            "tags": ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image"],
        },
        {
            "id": "WarriorMama777/OrangeMixs",
            "inference endpoint": [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2",
                "WarriorMama777/OrangeMixs",
                "andite/anything-v4.0",
                "prompthero/openjourney",
            ],
            "likes": 2439,
            "description": "\n\n\n\n",
            "tags": ["stable-diffusion", "text-to-image"],
        },
        {
            "id": "prompthero/openjourney",
            "inference endpoint": [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2",
                "WarriorMama777/OrangeMixs",
                "andite/anything-v4.0",
                "prompthero/openjourney",
            ],
            "likes": 2060,
            "description": "\n# Openjourney is an open source Stable Diffusion fine tuned model on Midjourney images, by [PromptH",
            "tags": ["stable-diffusion", "text-to-image"],
        },
        {
            "id": "andite/anything-v4.0",
            "inference endpoint": [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2",
                "WarriorMama777/OrangeMixs",
                "andite/anything-v4.0",
                "prompthero/openjourney",
            ],
            "likes": 1815,
            "description": "\n\nFantasy.ai is the official and exclusive hosted AI generation platform that holds a commercial use",
            "tags": [
                "stable-diffusion",
                "stable-diffusion-diffusers",
                "text-to-image",
                "diffusers",
            ],
        },
        {
            "id": "stabilityai/stable-diffusion-2",
            "inference endpoint": [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2",
                "WarriorMama777/OrangeMixs",
                "andite/anything-v4.0",
                "prompthero/openjourney",
            ],
            "likes": 1333,
            "description": "\n\n# Stable Diffusion v2 Model Card\nThis model card focuses on the model associated with the Stable D",
            "tags": ["stable-diffusion", "text-to-image"],
        },
    ]
