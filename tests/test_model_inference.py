import pytest
from dotenv import load_dotenv

from hugginggpt.model_inference import infer

load_dotenv()


@pytest.mark.skip(reason="huggingfacehub API not yet mocked")
def test_infer(model_id, data, task):
    result = infer(model_id=model_id, args=data, task=task)
    print(result)
    assert "generated image" in result.keys()


@pytest.fixture
def model_id():
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def task():
    return "text-to-image"


@pytest.fixture
def data():
    return {"text": "Draw me a sheep."}
