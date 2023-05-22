import json
import logging

from langchain import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import load_prompt

from hugginggpt.exceptions import ResponseGenerationException, wrap_exceptions
from hugginggpt.model_inference import TaskSummary
from hugginggpt.resources import get_prompt_resource, prepend_resource_dir

logger = logging.getLogger(__name__)


@wrap_exceptions(ResponseGenerationException, "Failed to generate assistant response")
def generate_response(
    user_input: str, task_summaries: list[TaskSummary], llm: BaseLLM
) -> str:
    """Use LLM agent to generate a response to the user's input, given task results."""
    logger.info("Starting response generation")
    sorted_task_summaries = sorted(task_summaries, key=lambda ts: ts.task.id)
    task_results_str = task_summaries_to_json(sorted_task_summaries)
    prompt_template = load_prompt(
        get_prompt_resource("response-generation-prompt.json")
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.predict(
        user_input=user_input, task_results=task_results_str, stop=["<im_end>"]
    )
    logger.info(f"Generated response: {response}")
    return response


def format_response(response: str) -> str:
    """Format the response to be more readable for user."""
    response = response.strip()
    response = prepend_resource_dir(response)
    return response


def task_summaries_to_json(task_summaries: list[TaskSummary]) -> str:
    dicts = [ts.dict() for ts in task_summaries]
    return json.dumps(dicts)
