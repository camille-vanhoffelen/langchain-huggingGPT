import copy
import logging

from pydantic import BaseModel, Field

from hugginggpt.exceptions import TaskParsingException, wrap_exceptions

logger = logging.getLogger(__name__)

GENERATED_TOKEN = "<GENERATED>"


class Task(BaseModel):
    # task and not name to aid prompt engineering
    task: str = Field(description="Name of the Machine Learning task")
    id: int = Field(description="ID of the task")
    dep: list[int] = Field(
        description="List of IDs of the tasks that this task depends on"
    )
    args: dict[str, str] = Field(description="Arguments for the task")

    # TODO add validation that doesn't break hacky dependency logic

    def depends_on_generated_resources(self):
        return self.dep != [-1] and any(
            GENERATED_TOKEN in v for v in self.args.values()
        )

    @wrap_exceptions(TaskParsingException, "Failed to replace generated resources")
    def replace_generated_resources(self, task_summaries: list):
        logger.info("Replacing generated resources")
        generated_resources = {
            k: parse_task_id(v) for k, v in self.args.items() if GENERATED_TOKEN in v
        }
        logger.info(
            f"Resources to replace, resource type -> task id: {generated_resources}"
        )
        for resource_type, task_id in generated_resources.items():
            matches = [
                v
                for k, v in task_summaries[task_id].inference_result.items()
                if k.startswith("generated " + resource_type)
            ]
            if len(matches) == 1:
                logger.info(
                    f"Match for generated {resource_type} in inference result of task {task_id}"
                )
                generated_resource = matches[0]
                logger.info(f"Replacing {resource_type} with {generated_resource}")
                self.args[resource_type] = generated_resource
                return self
            else:
                raise Exception(
                    f"Cannot find unique required generated {resource_type} in inference result of task {task_id}"
                )


class Tasks(BaseModel):
    __root__: list[Task] = Field(description="List of Machine Learning tasks")

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)


@wrap_exceptions(TaskParsingException, "Failed to parse tasks")
def parse_tasks(tasks_str) -> list[Task]:
    if tasks_str == "[]":
        raise ValueError("Task string empty, cannot parse")
    logger.info(f"Parsing tasks string: {tasks_str}")
    # TODO more complex logic here
    tasks_str = tasks_str.strip()
    # TODO Replace by PydanticOutputParser once fix list parsing
    tasks = Tasks.parse_raw(tasks_str)
    tasks = unfold(tasks.__root__)
    tasks = fix_dep(tasks)
    logger.info(f"Parsed tasks: {tasks}")
    return tasks


class TaskParsingException(Exception):
    pass


def parse_task_id(resource_str):
    return int(resource_str.split("-")[1])


# TODO Does this really remove all generated dependencies, and use the GENERATED tag ids instead?
# TODO Refactor
def fix_dep(tasks: list[Task]) -> list[Task]:
    for task in tasks:
        args = task.args
        task.dep = []
        for k, v in args.items():
            if "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task.dep:
                    task.dep.append(dep_task_id)
        if len(task.dep) == 0:
            task.dep = [-1]
    return tasks


def unfold(tasks: list[Task]) -> list[Task]:
    """Split tasks that have several generated resources folded into a single argument"""
    folded_tasks = (t for t in tasks if is_folded(t))
    split_tasks = (split(t) for t in folded_tasks)
    unfolded_tasks = [t for t in tasks if not is_folded(t)]
    # flatten
    split_tasks = [t for split_task in split_tasks for t in split_task]
    unfolded_tasks.extend(split_tasks)
    return unfolded_tasks


def split(task: Task):
    for key, value in task.args.items():
        if value.count(GENERATED_TOKEN) > 1:
            logger.debug(f"Unfolding task {task.id}")
            generated_items = value.split(",")
            for item in generated_items:
                new_task = copy.deepcopy(task)
                dep_task_id = int(item.split("-")[1])
                new_task.dep = [dep_task_id]
                new_task.args[key] = item.strip()
                yield new_task


def is_folded(task: Task) -> bool:
    for key, value in task.args.items():
        if value.count(GENERATED_TOKEN) > 1:
            logger.debug(f"Task {task.id} is folded")
            return True
    return False
