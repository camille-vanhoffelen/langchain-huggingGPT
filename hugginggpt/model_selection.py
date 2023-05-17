import asyncio
import json
import logging

import aiohttp
from langchain import LLMChain
from langchain.llms.base import BaseLLM
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import load_prompt
from pydantic import BaseModel, Field

from hugginggpt.exceptions import ModelSelectionException, async_wrap_exceptions
from hugginggpt.model_scraper import get_top_k_models
from hugginggpt.resources import get_prompt_resource
from hugginggpt.task_parsing import Task, Tasks

logger = logging.getLogger(__name__)


async def select_hf_models(
    user_input: str,
    tasks: Tasks,
    model_selection_llm: BaseLLM,
    output_fixing_llm: BaseLLM,
):
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            aio_tasks = []
            for task in tasks:
                aio_tasks.append(
                    tg.create_task(
                        select_model(
                            user_input=user_input,
                            task=task,
                            model_selection_llm=model_selection_llm,
                            output_fixing_llm=output_fixing_llm,
                            session=session,
                        )
                    )
                )
        results = await asyncio.gather(*aio_tasks)
        return {task_id: model for task_id, model in results}


# TODO implement extra tasks dependency parsing l.899
@async_wrap_exceptions(ModelSelectionException, "Failed to select model")
async def select_model(
    user_input: str,
    task: Task,
    model_selection_llm: BaseLLM,
    output_fixing_llm: BaseLLM,
    session: aiohttp.ClientSession,
):
    logger.info(f"Starting model selection for task: {task.task}")

    top_k_models = await get_top_k_models(
        task=task.task, top_k=5, max_description_length=100, session=session
    )

    if task.task in [
        "summarization",
        "translation",
        "conversational",
        "text-generation",
        "text2text-generation",
    ]:
        model = Model(
            id="openai",
            reason="Text generation tasks are best handled by OpenAI models",
        )
    else:
        prompt_template = load_prompt(
            get_prompt_resource("model-selection-prompt.json")
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=model_selection_llm)
        # Need to replace double quotes with single quotes for correct response generation
        task_str = task.json().replace('"', "'")
        models_str = json.dumps(top_k_models).replace('"', "'")
        output = await llm_chain.apredict(
            user_input=user_input, task=task_str, models=models_str, stop=["<im_end>"]
        )
        logger.debug(f"Model selection raw output: {output}")

        # TODO use apredict_and_parse instead above
        parser = PydanticOutputParser(pydantic_object=Model)
        fixing_parser = OutputFixingParser.from_llm(
            parser=parser, llm=output_fixing_llm
        )
        model = fixing_parser.parse(output)

    logger.info(f"For task: {task.task}, selected model: {model}")
    return task.id, model


class Model(BaseModel):
    id: str = Field(description="ID of the model")
    reason: str = Field(description="Reason for selecting this model")
    # TODO add validation that doesn't break hacky dependency logic
