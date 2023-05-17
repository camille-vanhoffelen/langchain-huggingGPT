import logging

from langchain import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import load_prompt

from hugginggpt.exceptions import TaskPlanningException, wrap_exceptions
from hugginggpt.history import ConversationHistory
from hugginggpt.resources import get_prompt_resource
from hugginggpt.task_parsing import parse_tasks

logger = logging.getLogger(__name__)


@wrap_exceptions(TaskPlanningException, "Failed to plan tasks")
def plan_tasks(user_input: str, history: ConversationHistory, llm: BaseLLM):
    logger.info("Starting task planning")
    task_planning_prompt_template = load_prompt(
        get_prompt_resource("task-planning-prompt.json")
    )
    llm_chain = LLMChain(prompt=task_planning_prompt_template, llm=llm)
    # TODO implement max length history truncation
    output = llm_chain.predict(
        user_input=user_input, history=history, stop=["<im_end>"]
    )
    logger.info(f"Task planning raw output: {output}")
    tasks = parse_tasks(output)
    return tasks
