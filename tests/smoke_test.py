import logging
import re

import aiohttp
from aioresponses import aioresponses
from responses import RequestsMock

from helpers.utils import AsyncFakeListLLM
from hugginggpt.llm_factory import LLMs
from main import standalone_mode

logger = logging.getLogger(__name__)


def main():
    _print_banner()
    logger.info("Starting smoke test")
    llms = create_llms()
    user_input = "Sheep are very cute. Based on the previous fact, answer the following question: Are sheep cute?"
    print("User:")
    print(user_input)
    print("Assistant:")
    with aioresponses() as ar, RequestsMock() as r:
        mock_model_status_responses(ar)
        mock_model_inference_responses(r)
        standalone_mode(
            user_input=user_input, llms=llms
        )
    logger.info("Smoke test complete")


def mock_model_status_responses(mocked_client):
    pattern = re.compile(r"^https://api-inference\.huggingface\.co/status/.*$")
    mocked_client.get(
        pattern,
        payload=dict(loaded="True"),
        repeat=True,
    )


def mock_model_inference_responses(mocked_client):
    pattern = re.compile(r"^https://api-inference\.huggingface\.co/models/.*$")
    mocked_client.post(
        pattern,
        json="Yes, sheep are very cute.",
    )


def create_llms():
    logger.info(f"Creating fake LLM")
    responses = [
        '[{"task": "question-answering", "id": 0, "dep": [-1], "args": {"question": "Are sheep cute?", "context": "Sheep are very cute."}}]'
    ]
    task_planning_llm = AsyncFakeListLLM(responses=responses)
    responses = [
        '{"id": "distilbert-base-cased-distilled-squad", "reason": "This model has the most potential to solve the user request as it has the highest number of likes and the most detailed description"}'
    ]
    model_selection_llm = AsyncFakeListLLM(responses=responses)
    responses = ["Yes, sheep are very cute."]
    model_inference_llm = AsyncFakeListLLM(responses=responses)
    responses = [
        "Based on the inference results, I can confidently say that sheep are indeed very cute."
    ]
    response_generation_llm = AsyncFakeListLLM(responses=responses)
    responses = ['{"output": "fixed"}']
    output_fixing_llm = AsyncFakeListLLM(responses=responses)
    return LLMs(
        task_planning_llm=task_planning_llm,
        model_selection_llm=model_selection_llm,
        model_inference_llm=model_inference_llm,
        response_generation_llm=response_generation_llm,
        output_fixing_llm=output_fixing_llm,
    )


def _print_banner():
    with open("resources/smoke-test-banner.txt", "r") as f:
        banner = f.read()
        logger.info("\n" + banner)


if __name__ == "__main__":
    main()
