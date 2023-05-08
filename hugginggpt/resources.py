import os

RESOURCES_DIR = "resources"
PROMPT_TEMPLATES_DIR = "prompt-templates"


def get_prompt_resource(prompt_name: str) -> str:
    return os.path.join(RESOURCES_DIR, PROMPT_TEMPLATES_DIR, prompt_name)
