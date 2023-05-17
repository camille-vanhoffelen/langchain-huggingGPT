import json

import pytest
from dotenv import load_dotenv
from langchain.llms.fake import FakeListLLM

from hugginggpt.model_inference import TaskSummary
from hugginggpt.model_selection import Model
from hugginggpt.response_generation import generate_response

load_dotenv()


def test_generate_response(question, task_summaries, llm, generated_response):
    response = generate_response(
        user_input=question, task_summaries=task_summaries, llm=llm
    )
    assert response == generated_response


@pytest.fixture
def question():
    return "Draw me a sheep."


@pytest.fixture
def task_summaries(task, model, inference_result):
    return [
        TaskSummary(
            task=task, model=model, inference_result=json.dumps(inference_result)
        )
    ]


@pytest.fixture
def task():
    return {
        "task": "text-to-image",
        "id": 0,
        "dep": [-1],
        "args": {"text": "Draw me a sheep."},
    }


@pytest.fixture
def inference_result():
    return {"generated image": "/images/sheep.png"}


@pytest.fixture
def model():
    return Model(id="great-project/great-model", reason="no particular reason")


@pytest.fixture
def generated_response():
    return "You asked to draw a sheep. I have drawn a sheep for you, and saved it at /images/sheep.png."


@pytest.fixture
def llm(generated_response):
    return FakeListLLM(responses=[generated_response])
