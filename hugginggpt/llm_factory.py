import logging
from collections import namedtuple

import tiktoken
from langchain import OpenAI

LLM_NAME = "text-davinci-003"
# Encoding for text-davinci-003
ENCODING_NAME = "p50k_base"
# Max input length for text-davinci-003
MAX_LENGTH = 4096

# As specified in huggingGPT paper
TASK_PLANNING_LOGIT_BIAS = 0.1
MODEL_SELECTION_LOGIT_BIAS = 5

logger = logging.getLogger(__name__)

LLMs = namedtuple(
    "LLMs",
    [
        "task_planning_llm",
        "model_selection_llm",
        "model_inference_llm",
        "response_generation_llm",
        "output_fixing_llm",
    ],
)


def create_llms():
    logger.info(f"Creating {LLM_NAME} LLM")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    task_parsing_highlight_ids = get_token_ids_for_task_parsing(encoding)
    choose_model_highlight_ids = get_token_ids_for_choose_model(encoding)

    task_planning_llm = OpenAI(
        model_name=LLM_NAME,
        temperature=0,
        logit_bias={
            token_id: TASK_PLANNING_LOGIT_BIAS
            for token_id in task_parsing_highlight_ids
        },
    )
    model_selection_llm = OpenAI(
        model_name=LLM_NAME,
        temperature=0,
        logit_bias={
            token_id: MODEL_SELECTION_LOGIT_BIAS
            for token_id in choose_model_highlight_ids
        },
    )
    model_inference_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    response_generation_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    output_fixing_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    return LLMs(
        task_planning_llm=task_planning_llm,
        model_selection_llm=model_selection_llm,
        model_inference_llm=model_inference_llm,
        response_generation_llm=response_generation_llm,
        output_fixing_llm=output_fixing_llm,
    )


def get_token_ids_for_task_parsing(encoding: tiktoken.Encoding):
    text = """{"task": "text-classification",  "token-classification", "text2text-generation", "summarization", "translation",  "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "visual-question-answering", "document-question-answering", "image-segmentation", "text-to-speech", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "args", "text", "path", "dep", "id", "<GENERATED>-"}"""
    res = encoding.encode(text)
    res = list(set(res))
    return res


def get_token_ids_for_choose_model(encoding: tiktoken.Encoding):
    text = """{"id": "reason"}"""
    res = encoding.encode(text)
    res = list(set(res))
    return res
