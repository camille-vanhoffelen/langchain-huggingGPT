import pytest

from hugginggpt.model_scraper import HUGGINGFACE_MODELS_MAP, filter_available_models


@pytest.mark.skip(reason="huggingfacehub API not yet mocked")
def test_filter_available_models(candidates):
    available_models = filter_available_models(candidates)


@pytest.fixture
def candidates(task, top_k):
    return HUGGINGFACE_MODELS_MAP[task][: top_k * 2]


@pytest.fixture
def task():
    return "text-generation"


@pytest.fixture
def top_k():
    return 5
