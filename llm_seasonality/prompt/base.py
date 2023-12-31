from abc import ABC, abstractmethod
from pydantic import BaseModel, computed_field, ConfigDict
from functools import cached_property
from datetime import datetime
from tqdm import tqdm
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_examples: int = 1
    dataset_name: DatasetEnum
    dataset_split: str
    dataset_revision: str
    dataset_outdir: str
    experiment_dt: None | datetime
    instruct_model: InstructEnum
    pipeline_kwargs: PipelineParams
    instruct_model_kwargs: ModelParams
    task_description: str
    date_experiment: bool = False

    col_accuracy: str = "accuracy"
    col_input: str = "input"
    col_output: str = "output"
    col_error: str = "error"
    col_output_token_len: str = "output_token_len"
    col_input_perplexity: str = "input_perplexity"
    col_output_perplexoty: str = "output_perplexity"

    @computed_field
    @cached_property
    def pipeline(self) -> transformers.Pipeline:
        pipeline_kwargs = dict(self.pipeline_kwargs)
        model_kwargs = dict(self.instruct_model_kwargs)
        return transformers.pipeline(model_kwargs=model_kwargs, **pipeline_kwargs)

    def load_dataset(self) -> datasets.Dataset:
        return datasets.load_dataset(
            self.dataset_name, self.dataset_revision, split=self.dataset_split
        )

    def load_annotated_dataset(self) -> datasets.Dataset:
        dataset = self.load_dataset()
        return dataset.map(self.format_prompt)

    def run(self):
        dataset = self.load_annotated_dataset()
        codegen = []
        print("Running pipeline")
        for out in tqdm(
            self.pipeline(
                KeyDataset(dataset, self.col_input),
                batch_size=self.pipeline_kwargs.batch_size,
            )
        ):
            codegen.append([x["generated_text"] for x in out])
        dataset.add_column(self.col_output, codegen)
        print("Executing code in a sandboxed container")
        dataset.map(self.run_program)
        print("Calculating accuracy")
        dataset.map(self.calc_accuracy)
        print("Calculating codegen len")
        dataset.map(self.calc_codegen_len)
        print("Calculating codegen length")
        dataset.map(self.calc_perplexity)
        print("Saving results to disk")
        dataset.save_to_disk()

    @abstractmethod
    def calc_accuracy(self, row) -> str:
        pass

    @abstractmethod
    def run_program(self, row) -> str:
        pass

    @abstractmethod
    def calc_codegen_len(self, row) -> str:
        pass

    @abstractmethod
    def calc_perplexity(self, row) -> str:
        pass

    @abstractmethod
    def format_example(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self, row) -> str:
        pass
