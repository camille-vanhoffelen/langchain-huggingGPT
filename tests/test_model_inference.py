import re

import pytest
import requests
import responses
from dotenv import load_dotenv

from hugginggpt.model_inference import infer
from hugginggpt.task_parsing import Task

load_dotenv()


def test_infer(task, model_id, mock_responses):
    with requests.Session() as session:
        result = infer(task=task, model_id=model_id, llm=None, session=session)
        assert result == "Yes, sheep are cute."


@pytest.fixture
def model_id():
    return "distilbert-base-cased-distilled-squad"


@pytest.fixture
def task(args):
    return Task(task="question-answering", id=0, dep=[-1], args=args)


@pytest.fixture
def args():
    return {"question": "Are sheep cute?", "context": "Sheep are very cute."}


@pytest.fixture
def mock_responses():
    with responses.RequestsMock() as r:
        pattern = re.compile(r"^https://api-inference\.huggingface\.co/models/.*$")
        r.post(
            pattern,
            json="Yes, sheep are cute.",
        )
        yield r
