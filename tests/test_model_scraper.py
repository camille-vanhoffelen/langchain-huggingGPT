import re

import aiohttp
import pytest
from aioresponses import aioresponses

from hugginggpt.model_scraper import HUGGINGFACE_MODELS_MAP, get_top_k_models
from hugginggpt.exceptions import ModelScrapingException


async def test_get_top_k_models(task, mock_models_loaded):
    async with aiohttp.ClientSession() as session:
        top_k_models_info = await get_top_k_models(
            task=task, top_k=1, max_description_length=10, session=session
        )
        assert len(top_k_models_info) == 1
        assert top_k_models_info[0]["id"] == HUGGINGFACE_MODELS_MAP[task][0]["id"]


async def test_models_not_loaded(task, mock_models_not_loaded):
    with pytest.raises(ModelScrapingException):
        async with aiohttp.ClientSession() as session:
            top_k_models_info = await get_top_k_models(
                task=task, top_k=1, max_description_length=10, session=session
            )
            assert len(top_k_models_info) == 1
            assert top_k_models_info[0]["id"] == HUGGINGFACE_MODELS_MAP[task][0]["id"]


@pytest.fixture
def task():
    return "text-to-image"


@pytest.fixture
def mock_models_loaded():
    with aioresponses() as m:
        mock_aioresponse(loaded=True, mocked=m)
        yield m


@pytest.fixture
def mock_models_not_loaded():
    with aioresponses() as m:
        mock_aioresponse(loaded=False, mocked=m)
        yield m


def mock_aioresponse(mocked, loaded: bool):
    pattern = re.compile(r"^https://api-inference\.huggingface\.co/status/.*$")
    mocked.get(
        pattern,
        payload=dict(loaded=loaded),
        repeat=True,
    )
