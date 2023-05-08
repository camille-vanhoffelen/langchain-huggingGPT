import pytest
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.prompts import load_prompt

from hugginggpt.response_generation import generate_response, sort_values
from hugginggpt.resources import get_prompt_resource

load_dotenv()


@pytest.mark.skip(reason="Unnecessary")
def test_response_generation_prompt(response_generation_prompt, question, results):
    results = sort_values(results)
    prompt_template = load_prompt(get_prompt_resource("response-generation-prompt.json"))
    prompt = prompt_template.format(user_input=question, results=results)
    assert prompt == response_generation_prompt


@pytest.mark.skip(reason="OpenAPI not yet mocked")
def test_generate_response(question, results):
    llm = OpenAI(temperature=0.0)
    response = generate_response(user_input=question, results=results, llm=llm)
    pass


@pytest.fixture
def response_generation_prompt():
    with open("tests/resources/response-generation.txt", "r") as f:
        return f.read()


@pytest.fixture
def question():
    return "Draw me a sheep."


@pytest.fixture
def results(task, model_selection_result, model_inference_result):
    return {
        0: {
            "task": task,
            "choose model result": model_selection_result,
            "inference result": model_inference_result,
        }
    }


@pytest.fixture
def task():
    return {
        "task": "text-to-image",
        "id": 0,
        "dep": [-1],
        "args": {"text": "Draw me a sheep."},
    }


@pytest.fixture
def id():
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def reason():
    return "This model has the most potential to solve the user request as it has the highest number of likes (6367) and the most detailed description, which suggests that it has the most potential to solve the user request"


@pytest.fixture
def model_selection_result(id, reason):
    # TODO why string and not dict here?
    return f'{{"id": "{id}", "reason": "{reason}"}}'


@pytest.fixture
def model_inference_result():
    return {"generated image": "/images/1b5e.png"}
