import re

import aiohttp
import pytest
from aioresponses import aioresponses
from dotenv import load_dotenv

from helpers.utils import AsyncFakeListLLM
from hugginggpt.exceptions import ModelSelectionException
from hugginggpt.model_selection import Model, select_model
from hugginggpt.task_parsing import Task

load_dotenv()


async def test_select_model(
    mock_aioresponse,
    user_input,
    task,
    model_selection_llm,
    output_fixing_llm,
    expected_model,
):
    async with aiohttp.ClientSession() as session:
        task_id, model = await select_model(
            user_input=user_input,
            task=task,
            model_selection_llm=model_selection_llm,
            output_fixing_llm=output_fixing_llm,
            session=session,
        )
    assert task_id == task.id
    assert model == expected_model


async def test_output_fixing(
    mock_aioresponse,
    user_input,
    task,
    faulty_model_selection_llm,
    output_fixing_llm,
    expected_model,
):
    async with aiohttp.ClientSession() as session:
        task_id, model = await select_model(
            user_input=user_input,
            task=task,
            model_selection_llm=faulty_model_selection_llm,
            output_fixing_llm=output_fixing_llm,
            session=session,
        )
    assert task_id == task.id
    assert model == expected_model


async def test_faulty_output_fixing(
    mock_aioresponse,
    user_input,
    task,
    faulty_model_selection_llm,
    faulty_output_fixing_llm,
    expected_model,
):
    with pytest.raises(ModelSelectionException):
        async with aiohttp.ClientSession() as session:
            await select_model(
                user_input=user_input,
                task=task,
                model_selection_llm=faulty_model_selection_llm,
                output_fixing_llm=faulty_output_fixing_llm,
                session=session,
            )


@pytest.fixture
def id():
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def reason():
    return "This model has the most potential to solve the user request as it has the highest number of likes and the most detailed description"


@pytest.fixture
def model_selection_response(id, reason):
    return f'{{"id": "{id}", "reason": "{reason}"}}'


@pytest.fixture
def model_selection_llm(model_selection_response):
    return AsyncFakeListLLM(responses=[model_selection_response])


@pytest.fixture
def output_fixing_llm(model_selection_response):
    return AsyncFakeListLLM(responses=[model_selection_response])


@pytest.fixture
def faulty_model_selection_response(id, reason):
    return f'{{"id": "{id}", "reason": "{reason} and also here are some random quotes that will mess up your json: "mouhahahah""}}'


@pytest.fixture
def faulty_model_selection_llm(faulty_model_selection_response):
    return AsyncFakeListLLM(responses=[faulty_model_selection_response])


@pytest.fixture
def faulty_output_fixing_llm(faulty_model_selection_response):
    return AsyncFakeListLLM(responses=[faulty_model_selection_response])


@pytest.fixture
def expected_model(id, reason):
    return Model(id=id, reason=reason)


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


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        pattern = re.compile(r"^https://api-inference\.huggingface\.co/status/.*$")
        m.get(
            pattern,
            payload=dict(loaded="True"),
            repeat=True,
        )
        yield m
