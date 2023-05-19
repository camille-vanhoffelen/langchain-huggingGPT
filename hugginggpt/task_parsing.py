import copy
import logging

from pydantic import BaseModel, Field

from hugginggpt.exceptions import TaskParsingException, wrap_exceptions

logger = logging.getLogger(__name__)

GENERATED_TOKEN = "<GENERATED>"


class Task(BaseModel):
    # This field is called 'task' and not 'name' to help with prompt engineering
    task: str = Field(description="Name of the Machine Learning task")
    id: int = Field(description="ID of the task")
    dep: list[int] = Field(
        description="List of IDs of the tasks that this task depends on"
    )
    args: dict[str, str] = Field(description="Arguments for the task")

    def depends_on_generated_resources(self) -> bool:
        """Returns True if the task args contains <GENERATED> placeholder tokens, False otherwise"""
        return self.dep != [-1] and any(
            GENERATED_TOKEN in v for v in self.args.values()
        )

    @wrap_exceptions(TaskParsingException, "Failed to replace generated resources")
    def replace_generated_resources(self, task_summaries: list):
        """Replaces <GENERATED> placeholder tokens in args with the generated resources from the task summaries"""
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
def parse_tasks(tasks_str: str) -> list[Task]:
    """Parses tasks from task planning json string"""
    if tasks_str == "[]":
        raise ValueError("Task string empty, cannot parse")
    logger.info(f"Parsing tasks string: {tasks_str}")
    tasks_str = tasks_str.strip()
    # Cannot use PydanticOutputParser because it fails when parsing top level list JSON string
    tasks = Tasks.parse_raw(tasks_str)
    # __root__ extracts list[Task] from Tasks object
    tasks = unfold(tasks.__root__)
    tasks = fix_dependencies(tasks)
    logger.info(f"Parsed tasks: {tasks}")
    return tasks


def parse_task_id(resource_str: str) -> int:
    """Parse task id from generated resource string, e.g. <GENERATED>-4 -> 4"""
    return int(resource_str.split("-")[1])


def fix_dependencies(tasks: list[Task]) -> list[Task]:
    """Ignores parsed tasks dependencies, and instead infers from task arguments"""
    for task in tasks:
        task.dep = infer_deps_from_args(task)
    return tasks


def infer_deps_from_args(task: Task) -> list[int]:
    """If GENERATED arg value, add to list of unique deps. If none, deps = [-1]"""
    deps = [parse_task_id(v) for v in task.args.values() if GENERATED_TOKEN in v]
    if not deps:
        deps = [-1]
    # deduplicate
    return list(set(deps))


def unfold(tasks: list[Task]) -> list[Task]:
    """A folded task has several generated resources folded into a single argument"""
    unfolded_tasks = []
    for task in tasks:
        folded_args = find_folded_args(task)
        if folded_args:
            unfolded_tasks.extend(split(task, folded_args))
        else:
            unfolded_tasks.append(task)
    return unfolded_tasks


def split(task: Task, folded_args: tuple[str, str]) -> list[Task]:
    """Split folded task into two same tasks, but separated generated resource arguments"""
    key, value = folded_args
    generated_items = value.split(",")
    split_tasks = []
    for item in generated_items:
        new_task = copy.deepcopy(task)
        dep_task_id = parse_task_id(item)
        new_task.dep = [dep_task_id]
        new_task.args[key] = item.strip()
        split_tasks.append(new_task)
    return split_tasks


def find_folded_args(task: Task) -> tuple[str, str] | None:
    """Finds folded args, e.g: 'image': '<GENERATED>-1,<GENERATED>-2'"""
    for key, value in task.args.items():
        if value.count(GENERATED_TOKEN) > 1:
            logger.debug(f"Task {task.id} is folded")
            return key, value
    return None
