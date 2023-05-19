import pytest
from langchain.llms.fake import FakeListLLM

from hugginggpt.history import ConversationHistory
from hugginggpt.task_parsing import Task
from hugginggpt.task_planning import plan_tasks, truncate_history


def test_plan_tasks(user_input, history, llm, expected_tasks):
    tasks = plan_tasks(user_input, history, llm)
    assert tasks == expected_tasks


def test_truncate_history(long_history: ConversationHistory):
    history = truncate_history(long_history)
    # 156 is max allowed messages for this long_history and MAX_HISTORY_TOKENS
    assert len(history) == 156
    # check last message is most recent
    n_rounds = int(len(long_history)/2)
    assert history[-1]["content"] == f"This is round of conversation number: {n_rounds}"


def test_do_not_truncate_history(short_history: ConversationHistory):
    history = truncate_history(short_history)
    assert len(history) == len(short_history)
    # check last message is most recent
    n_rounds = int(len(short_history)/2)
    assert history[-1]["content"] == f"This is round of conversation number: {n_rounds}"


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
def long_history():
    return create_conversation(1000)


@pytest.fixture
def short_history():
    return create_conversation(10)


def create_conversation(n_rounds: int) -> ConversationHistory:
    history = ConversationHistory()
    for i in range(n_rounds):
        history.add(role="user", content="Which round of conversation is this?")
        history.add(role="assistant", content=f"This is round of conversation number: {i + 1}")
    return history


@pytest.fixture
def llm(response):
    return FakeListLLM(responses=[response])


@pytest.fixture
def expected_tasks():
    return [
        Task(task="text-to-image", id=0, dep=[-1], args={"text": "Draw me a sheep."})
    ]
