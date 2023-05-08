import json
import logging

from langchain import LLMChain
from langchain.prompts import load_prompt

from hugginggpt.exceptions import ResponseGenerationException, wrap_exceptions
from hugginggpt.resources import get_prompt_resource
from hugginggpt.task_parsing import TaskSummary

logger = logging.getLogger(__name__)


@wrap_exceptions(ResponseGenerationException, "Failed to generate assistant response")
def generate_response(user_input: str, task_summaries: dict[int, TaskSummary], llm):
    logger.info("Starting response generation")
    # TODO find better name than results
    sorted_task_summaries = sort_values(task_summaries)
    results = [ts.dict() for ts in sorted_task_summaries]
    results_str = json.dumps(results)
    prompt_template = load_prompt(
        get_prompt_resource("response-generation-prompt.json")
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.predict(
        user_input=user_input, results=results_str, stop=["<im_end>"]
    )
    logger.info(f"Generated response: {response}")
    return response


def sort_values(d):
    return [v for k, v in sorted(d.items(), key=lambda item: item[0])]
