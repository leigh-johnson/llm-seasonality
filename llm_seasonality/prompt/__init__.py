import importlib
from typing import Dict, Any
from llm_seasonality.prompt.base import BaseTask


def load_prompt(task: str, task_kwargs: Dict[str, Any]) -> BaseTask:
    prompt_module = importlib.import_module(f"llm_seasonality.prompt.{task}")
    prompt = prompt_module.PROMPT(**task_kwargs)
    return prompt
