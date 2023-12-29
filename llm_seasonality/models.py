from enum import Enum
from pydantic import BaseModel, ConfigDict
import transformers


class InstructEnum(str, Enum):
    """
    Models fine-tuned for intruction-following
    """

    LLAMA2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
    CODELLAMA_7B_INSTRUCT_HF = "codellama/CodeLlama-7b-Instruct-hf"
    CODELLAMA_7B_PYTHON_HF = "codellama/CodeLlama-7b-Python-hf"


class DatasetEnum(str, Enum):
    GSM8K = "gsm8k"


class ModelParams(BaseModel):
    do_sample: bool
    max_length: int
    temperature: float
    top_p: float


class PipelineParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    eos_token_id: int
    max_length: int
    batch_size: int
    device_map: str = "auto"
    task: str = "text-generation"
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    num_return_sequences: int = 1
