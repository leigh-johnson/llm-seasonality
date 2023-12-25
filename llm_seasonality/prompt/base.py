from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datasets import Dataset

from llm_seasonality.models import InstructEnum, DatasetEnum

ANSWER_TOKEN = "####"  # indictates the final answer in ground truth


class BasePrompt(BaseModel, ABC):
    """
    This class is responsible for prompt formatting
    """

    num_examples: int = 1
    dataset: Dataset
    instruct_model: InstructEnum
    task_description: str

    @abstractmethod
    def format_example(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self) -> str:
        pass
