import json

import pytest

from hugginggpt.model_selection import Model
from hugginggpt.task_parsing import GENERATED_TOKEN, Task, parse_tasks
from hugginggpt.model_inference import TaskSummary


def test_parse_tasks(tasks_str):
    tasks = parse_tasks(tasks_str)
    assert len(tasks) == 1
    assert len(tasks[0].dict()) == 4


def test_parse_complex_tasks(complex_tasks_str):
    tasks = parse_tasks(complex_tasks_str)
    assert len(tasks) == 4


def test_parse_bad_tasks(bad_tasks_str):
    with pytest.raises(Exception):
        parse_tasks(bad_tasks_str)


def test_unfold_tasks(folded_tasks_str):
    tasks = parse_tasks(folded_tasks_str)
    assert len(tasks) == 2
    deps = [t.dep for t in tasks]
    assert [0] in deps
    assert [1] in deps
    arg_values = [v for t in tasks for v in t.args.values()]
    assert GENERATED_TOKEN + "-0" in arg_values
    assert GENERATED_TOKEN + "-1" in arg_values


def test_fix_task_dependencies_when_mismatch(mismatch_dependencies_task_str):
    tasks = parse_tasks(mismatch_dependencies_task_str)
    assert len(tasks) == 2
    deps = [t.dep for t in tasks]
    assert [0] in deps
    assert [1] in deps

def test_fix_task_dependencies_when_extra(extra_dependencies_task_str):
    tasks = parse_tasks(extra_dependencies_task_str)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.dep == [-1]


def test_does_not_depend_on_generated_resources(task):
    assert not task.depends_on_generated_resources()


def test_depends_on_generated_resources(dependent_task):
    assert dependent_task.depends_on_generated_resources()


def test_replace_generated_resources(dependent_task, task_summaries):
    dependent_task.replace_generated_resources(task_summaries)
    assert dependent_task.args["image"] == "/images/007.png"


def test_do_not_replace_generated_resources(task, task_summaries):
    original_args = task.args.copy()
    task.replace_generated_resources(task_summaries)
    assert task.args == original_args


@pytest.fixture
def tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }}]'


@pytest.fixture
def complex_tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Generate an image of two bulls" }}, {"task": "object-detection", "id": 1, "dep": [-1], "args": {"image": "<GENERATED>-0" }}, {"task": "visual-question-answering", "id": 2, "dep": [1], "args": {"image": "<GENERATED>-1", "text": "How many horns in the image"}}, {"task": "text-generation", "id": 3, "dep": [2], "args": {"text": "<GENERATED>-2" }}]'


@pytest.fixture
def bad_tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }]'


@pytest.fixture
def folded_tasks_str():
    return '[{"task": "image-to-text", "id": 2, "dep": [0, 1], "args": {"image": "<GENERATED>-0, <GENERATED>-1" }}]'


@pytest.fixture
def mismatch_dependencies_task_str():
    return '[{"task": "image-to-text", "id": 2, "dep": [2, 3], "args": {"image": "<GENERATED>-0, <GENERATED>-1" }}]'


@pytest.fixture
def extra_dependencies_task_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [2, 3], "args": {"text": "Draw me a sheep." }}]'


@pytest.fixture
def task():
    return Task(task="text-to-image", id=0, dep=[-1], args={"text": "Draw me a sheep."})


@pytest.fixture
def dependent_task():
    return Task(
        task="object-detection",
        id=1,
        dep=[0],
        args={"image": f"{GENERATED_TOKEN}-0"},
    )


@pytest.fixture
def model():
    return Model(id="great-model", reason="no particular reason")


@pytest.fixture
def inference_result():
    return json.dumps({"generated image": "/images/007.png"})


@pytest.fixture
def task_summaries(dependent_task, inference_result, model):
    return {
        0: TaskSummary(
            task=dependent_task, inference_result=inference_result, model=model
        )
    }
