import json
import logging

from langchain import LLMChain, OpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import load_prompt
from pydantic import BaseModel, Field

from hugginggpt.exceptions import ModelSelectionException, wrap_exceptions
from hugginggpt.resources import get_prompt_resource

logger = logging.getLogger(__name__)


# TODO implement multithreading
# TODO implement extra tasks dependency parsing l.899
@wrap_exceptions(ModelSelectionException, "Failed to select model")
def select_model(user_input, task, top_k_models, llm):
    logger.info("Starting model selection")

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
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)
        # Need to replace double quotes with single quotes for correct response generation
        task_str = task.json().replace('"', "'")
        models_str = json.dumps(top_k_models).replace('"', "'")
        output = llm_chain.predict(
            user_input=user_input, task=task_str, models=models_str, stop=["<im_end>"]
        )
        logger.debug(f"Model selection raw output: {output}")

        parser = PydanticOutputParser(pydantic_object=Model)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=OpenAI())
        model = fixing_parser.parse(output)

    logger.info(f"Selected model: {model}")
    return model


class Model(BaseModel):
    id: str = Field(description="ID of the model")
    reason: str = Field(description="Reason for selecting this model")
    # TODO add validation that doesn't break hacky dependency logic


class ModelSelectionException(Exception):
    pass
