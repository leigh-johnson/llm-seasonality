from abc import ABC, abstractmethod
from pydantic import BaseModel
import datasets
import transformers

from transformers.pipelines.pt_utils import KeyDataset

from llm_seasonality.models import (
    InstructEnum,
    DatasetEnum,
    PipelineParams,
    ModelParams,
)

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
    pipeline_kwargs: PipelineParams
    instruct_model_kwargs: ModelParams
    task_description: str

    col_accuracy: str = "accuracy"
    col_prompt: str = "prompt"
    col_codegen: str = "codegen"
    col_stdout: str = "stdout"
    col_stderr: str = "stderr"

    def load_dataset(self) -> datasets.Dataset:
        return datasets.load_dataset(
            self.dataset_name, self.dataset_revision, split=self.dataset_split
        )

    def load_annotated_dataset(self) -> datasets.Dataset:
        dataset = self.load_dataset()
        return dataset.map(self.format_prompt)

    def load_pipeline(self) -> transformers.Pipeline:
        pipeline_kwargs = dict(self.pipeline_kwargs)
        model_kwargs = dict(self.instruct_model_kwargs)
        return transformers.pipeline(model_kwargs=model_kwargs, **pipeline_kwargs)

    def run(self):
        dataset = self.load_annotated_dataset()
        pipe = self.load_pipeline()
        for out in pipe(
            KeyDataset(dataset, self.col_prompt),
            batch_size=self.pipeline_kwargs.batch_size,
        ):
            print(out)

    @abstractmethod
    def calc_accuracy(self, row) -> str:
        pass

    @abstractmethod
    def format_example(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self, row) -> str:
        pass
