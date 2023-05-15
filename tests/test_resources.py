import pytest

import hugginggpt.resources as resources


def test_encode_image(mock_get_resource_url, image_arg):
    image_data = resources.encode_image(image_arg)
    assert type(image_data) == bytes
    assert len(image_data) == 448026


def test_encode_audio(mock_get_resource_url, audio_arg):
    audio_data = resources.encode_audio(audio_arg)
    assert type(audio_data) == bytes
    assert len(audio_data) == 158720


@pytest.fixture
def image_arg():
    return "/images/1193.png"


@pytest.fixture
def audio_arg():
    return "/audios/b6b7.flac"


@pytest.fixture
def mock_get_resource_url(monkeypatch):
    def mock_fn(url):
        return "tests/resources" + url

    monkeypatch.setattr(resources, "get_resource_url", mock_fn)
