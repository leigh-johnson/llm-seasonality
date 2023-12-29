from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel
import datasets


from llm_seasonality.models import InstructEnum, DatasetEnum

ANSWER_TOKEN = "####"  # indictates the final answer in ground truth


class BasePrompt(BaseModel, ABC):
    """
    This class is responsible for prompt formatting
    """

    num_examples: int = 1
    dataset_name: DatasetEnum
    dataset_split: str
    dataset_revision: str
    instruct_model: InstructEnum
    task_description: str

    def load_dataset(self) -> datasets.Dataset:
        return datasets.load(
            self.dataset_name, self.dataset_revision, split=self.dataset_split
        )

    @abstractmethod
    def format_example(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self, row) -> str:
        pass
