import asyncio
import json
import logging
import os
from collections import defaultdict

from aiohttp import ClientSession

from hugginggpt.exceptions import ModelScrapingException, wrap_exceptions

logger = logging.getLogger(__name__)


def read_huggingface_models_metadata():
    with open("resources/huggingface_models_metadata.jsonl") as f:
        models = [json.loads(line) for line in f]
    models_map = defaultdict(list)
    for model in models:
        models_map[model["task"]].append(model)
    return models_map


# TODO place somewhere else
HUGGINGFACE_MODELS_MAP = read_huggingface_models_metadata()
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}


@wrap_exceptions(ModelScrapingException, "Failed to find compatible models")
def get_top_k_models(task, top_k, max_description_length):
    # Number of potential candidates changed from top 10 to top_k*2
    candidates = HUGGINGFACE_MODELS_MAP[task][: top_k * 2]
    logger.debug(f"Task: {task}; All candidate models: {[c['id'] for c in candidates]}")
    available_models = asyncio.run(filter_available_models(candidates=candidates))
    logger.debug(
        f"Task: {task}; Available models: {[c['id'] for c in available_models]}"
    )
    top_k_available_models = available_models[:top_k]
    if not top_k_available_models:
        raise Exception(f"No available models for task: {task}")
    logger.debug(
        f"Task: {task}; Top {top_k} available models: {[c['id'] for c in top_k_available_models]}"
    )
    top_k_models_info = [
        {
            "id": model["id"],
            "likes": model.get("likes"),
            "description": model.get("description", "")[:max_description_length],
            "tags": model.get("meta").get("tags") if model.get("meta") else None,
        }
        for model in top_k_available_models
    ]
    return top_k_models_info


async def filter_available_models(candidates):
    async with ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(model_status(model_id=c["id"], session=session)) for c in candidates]
        return await asyncio.gather(*tasks)


async def model_status(model_id: str, session: ClientSession) -> tuple[str, bool]:
    url = f"https://api-inference.huggingface.co/status/{model_id}"
    r = await session.get(url, headers=HUGGINGFACE_HEADERS)
    # TODO remove
    status = r.status
    json_response = await r.json()
    logger.debug("Model status response: %s", json_response)
    return (model_id, True) if model_is_available(status=status, json_response=json_response) else (model_id, False)


def model_is_available(status: int, json_response: dict[str, any]):
    return (
            status == 200
            and "loaded" in json_response
            and json_response["loaded"]
    )
