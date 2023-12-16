from enum import Enum


class InstructModel(str, Enum):
    """
    Models fine-tuned for intruction-following
    """

    LLAMA2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
    CODELLAMA_7B_INSTRUCT_HF = "codellama/CodeLlama-7b-Instruct-hf"
    CODELLAMA_7B_PYTHON_HF = "codellama/CodeLlama-7b-Python-hf"
