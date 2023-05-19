import asyncio
import json
import logging

import click
from dotenv import load_dotenv

from hugginggpt import generate_response, infer, plan_tasks
from hugginggpt.history import ConversationHistory
from hugginggpt.llm_factory import LLMs, create_llms
from hugginggpt.log import setup_logging
from hugginggpt.model_inference import TaskSummary
from hugginggpt.model_selection import select_hf_models

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@click.command()
@click.option("-p", "--prompt", type=str, help="Prompt for huggingGPT")
def main(prompt):
    _print_banner()
    llms = create_llms()
    if prompt:
        standalone_mode(user_input=prompt, llms=llms)

    else:
        interactive_mode(llms=llms)


def standalone_mode(user_input: str, llms: LLMs) -> str:
    try:
        response = _compute(
            user_input=user_input,
            history=ConversationHistory(),
            llms=llms,
        )
        print(response.strip())
        return response
    except Exception as e:
        logger.exception("")
        print(
            f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
        )


def interactive_mode(llms: LLMs):
    print("Please enter your request. End the conversation with 'exit'")
    history = ConversationHistory()
    while True:
        try:
            user_input = click.prompt("User")
            if user_input.lower() == "exit":
                break

            logger.info(f"User input: {user_input}")
            response = _compute(
                user_input=user_input,
                history=history,
                llms=llms,
            )
            print(f"Assistant:{response}")

            history.add(role="user", content=user_input)
            history.add(role="assistant", content=response)
        except Exception as e:
            logger.exception("")
            print(
                f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
            )


def _compute(
    user_input: str,
    history: ConversationHistory,
    llms: LLMs,
) -> str:
    tasks = plan_tasks(
        user_input=user_input, history=history, llm=llms.task_planning_llm
    )

    sorted(tasks, key=lambda t: max(t.dep))
    logger.info(f"Sorted tasks: {tasks}")

    hf_models = asyncio.run(
        select_hf_models(
            user_input=user_input,
            tasks=tasks,
            model_selection_llm=llms.model_selection_llm,
            output_fixing_llm=llms.output_fixing_llm,
        )
    )

    task_summaries = []
    for task in tasks:
        logger.info(f"Starting task: {task}")
        if task.depends_on_generated_resources():
            task = task.replace_generated_resources(task_summaries=task_summaries)
        model = hf_models[task.id]
        inference_result = infer(
            task=task, model_id=model.id, llm=llms.model_inference_llm
        )
        task_summaries.append(
            TaskSummary(
                task=task, model=model, inference_result=json.dumps(inference_result)
            )
        )
        logger.info(f"Finished task: {task}")
    logger.info("Finished all tasks")
    logger.debug(f"Task summaries: {task_summaries}")

    response = generate_response(
        user_input=user_input,
        task_summaries=task_summaries,
        llm=llms.response_generation_llm,
    )
    return response


def _print_banner():
    with open("resources/banner.txt", "r") as f:
        banner = f.read()
        logger.info("\n" + banner)


if __name__ == "__main__":
    main()
