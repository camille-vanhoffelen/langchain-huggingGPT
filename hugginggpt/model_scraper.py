import json
import logging
import os
from collections import defaultdict
from multiprocessing import Pool

import requests

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
    all_available_models = filter_available_models(candidates)
    logger.debug(
        f"Task: {task}; Available models: {[c['id'] for c in all_available_models]}"
    )
    top_k_available_models = all_available_models[:top_k]
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


def filter_available_models(candidates):
    all_model_ids = [c["id"] for c in candidates]
    with Pool() as p:
        results = p.map(check_huggingface_model_status, all_model_ids)
    available_model_ids = [model_id for model_id, status in results if status]
    return [c for c in candidates if c["id"] in available_model_ids]


def check_huggingface_model_status(model_id):
    url = f"https://api-inference.huggingface.co/status/{model_id}"
    r = requests.get(url, headers=HUGGINGFACE_HEADERS)
    return (model_id, True) if model_is_available(r) else (model_id, False)


def model_is_available(response):
    return (
        response.status_code == 200
        and "loaded" in response.json()
        and response.json()["loaded"]
    )
