import pytest
from langchain.llms.fake import FakeListLLM

from hugginggpt.history import ConversationHistory
from hugginggpt.task_parsing import Task, Tasks
from hugginggpt.task_planning import plan_tasks


def test_plan_tasks(user_input, history, llm, expected_tasks):
    tasks = plan_tasks(user_input, history, llm)
    # assert tasks == expected_tasks
    pass


@pytest.fixture
def user_input():
    return "Draw me a sheep."


@pytest.fixture
def response():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }}]'


@pytest.fixture
def history():
    return ConversationHistory()


@pytest.fixture
def llm(response):
    return FakeListLLM(responses=[response])


@pytest.fixture
def expected_tasks():
    return Tasks(
        __root__=[
            Task(
                task="text-to-image", id=0, dep=[-1], args={"text": "Draw me a sheep."}
            )
        ]
    )
