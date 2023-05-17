import logging
from collections import namedtuple
from typing import List, Optional

from langchain import OpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
)
from langchain.llms.fake import FakeListLLM

from hugginggpt.get_token_ids import (
    get_token_ids_for_choose_model,
    get_token_ids_for_task_parsing,
)

# TODO implement ChatGPT
TEXT_DAVINCI_003 = "text-davinci-003"
FAKE_LLM = "fake-LLM"
MODEL_CHOICES = [TEXT_DAVINCI_003, FAKE_LLM]

logger = logging.getLogger(__name__)

LLMs = namedtuple(
    "LLMs",
    [
        "task_planning_llm",
        "model_selection_llm",
        "response_generation_llm",
        "output_fixing_llm",
    ],
)


# TODO remove support for multi models
def create_llms(llm_type: str):
    # TODO better loading of logit biases
    if llm_type == TEXT_DAVINCI_003:
        logger.info(f"Creating {TEXT_DAVINCI_003} LLM")
        task_parsing_highlight_ids = get_token_ids_for_task_parsing(llm_type)
        choose_model_highlight_ids = get_token_ids_for_choose_model(llm_type)
        task_planning_llm = OpenAI(
            model_name=TEXT_DAVINCI_003,
            temperature=0,
            logit_bias={token_id: 0.1 for token_id in task_parsing_highlight_ids},
        )
        model_selection_llm = OpenAI(
            model_name=TEXT_DAVINCI_003,
            temperature=0,
            logit_bias={token_id: 5 for token_id in choose_model_highlight_ids},
        )
        response_generation_llm = OpenAI(model_name=TEXT_DAVINCI_003, temperature=0)
        output_fixing_llm = OpenAI(model_name=TEXT_DAVINCI_003, temperature=0)
        return LLMs(
            task_planning_llm=task_planning_llm,
            model_selection_llm=model_selection_llm,
            response_generation_llm=response_generation_llm,
            output_fixing_llm=output_fixing_llm,
        )

    if llm_type == FAKE_LLM:
        logger.info(f"Creating {FAKE_LLM} LLM")
        responses = [
            '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }}]'
        ]
        task_planning_llm = FakeListLLM(responses=responses)
        responses = [
            '{"id": "runwayml/stable-diffusion-v1-5", "reason": "This model has the most potential to solve the user request as it has the highest number of likes and the most detailed description"}'
        ]
        model_selection_llm = FakeListLLM(responses=responses)
        responses = [
            "I understand your request. Based on the inference results, I have generated an image of a sheep for you. The image is located at the following URL: /images/1b5e.png. I used the model runwayml/stable-diffusion-v1-5 to generate the image. This model has the most potential to solve the user request as it has the highest number of likes (6367) and the most detailed description, which suggests that it has the most potential to solve the user request. I hope this image meets your expectations. Is there anything else I can help you with?"
        ]
        response_generation_llm = FakeListLLM(responses=responses)
        responses = ['{"output": "fixed"}']
        output_fixing_llm = FakeListLLM(responses=responses)
        return LLMs(
            task_planning_llm=task_planning_llm,
            model_selection_llm=model_selection_llm,
            response_generation_llm=response_generation_llm,
            output_fixing_llm=output_fixing_llm,
        )


class AsyncFakeListLLM(FakeListLLM):
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i]
        self.i += 1
        return response
