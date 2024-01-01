import importlib
from typing import Dict, Any
from llm_seasonality.prompt.base import BasePrompt


def load_prompt(task, **kwargs) -> BasePrompt:
    prompt_module = importlib.import_module(f"llm_seasonality.prompt.{task}")
    prompt = prompt_module.PROMPT(**kwargs)
    return prompt
