import asyncio
import json
import logging
from collections import defaultdict

from aiohttp import ClientSession

from hugginggpt.exceptions import ModelScrapingException, async_wrap_exceptions
from hugginggpt.huggingface_api import (
    HUGGINGFACE_HEADERS,
    HUGGINGFACE_INFERENCE_API_STATUS_URL,
)

logger = logging.getLogger(__name__)


def read_huggingface_models_metadata():
    """Reads the metadata of all huggingface models from the local models cache file."""
    with open("resources/huggingface-models-metadata.jsonl") as f:
        models = [json.loads(line) for line in f]
    models_map = defaultdict(list)
    for model in models:
        models_map[model["task"]].append(model)
    return models_map


HUGGINGFACE_MODELS_MAP = read_huggingface_models_metadata()


@async_wrap_exceptions(ModelScrapingException, "Failed to find compatible models already loaded in the huggingface inference API.")
async def get_top_k_models(
    task: str, top_k: int, max_description_length: int, session: ClientSession
):
    """Returns the best k available huggingface models for a given task, sorted by number of likes."""
    # Number of potential candidates changed from top 10 to top_k*2
    candidates = HUGGINGFACE_MODELS_MAP[task][: top_k * 2]
    logger.debug(f"Task: {task}; All candidate models: {[c['id'] for c in candidates]}")
    available_models = await filter_available_models(
        candidates=candidates, session=session
    )
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


async def filter_available_models(candidates, session: ClientSession):
    """Filters out models that are not available or loaded in the huggingface inference API.
    Runs concurrently."""
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(model_status(model_id=c["id"], session=session))
            for c in candidates
        ]
    results = await asyncio.gather(*tasks)
    available_model_ids = [model_id for model_id, status in results if status]
    return [c for c in candidates if c["id"] in available_model_ids]


async def model_status(model_id: str, session: ClientSession) -> tuple[str, bool]:
    url = HUGGINGFACE_INFERENCE_API_STATUS_URL + model_id
    r = await session.get(url, headers=HUGGINGFACE_HEADERS)
    status = r.status
    json_response = await r.json()
    logger.debug(f"Model {model_id} status: {status}, response: {json_response}")
    return (
        (model_id, True)
        if model_is_available(status=status, json_response=json_response)
        else (model_id, False)
    )


def model_is_available(status: int, json_response: dict[str, any]):
    return status == 200 and "loaded" in json_response and json_response["loaded"]
