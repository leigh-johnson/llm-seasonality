import click
import os
from llm_seasonality.models import InstructModel


@click.command()
@click.option("--cache-dir", default="/mnt/spindle/.tmp")
@click.option("--dataset", type=click.Choice(["gsm8k"], case_sensitive=False))
@click.option("--dataset-split", type=str, default="test")
@click.option(
    "--instruct-model",
    type=click.Choice(
        InstructModel,
        case_sensitive=False,
    ),
    default=InstructModel.LLAMA2_7B_CHAT_HF,
)
@click.option(
    "--max-length",
    type=int,
    default=4096,
    help="https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens",
)
@click.option(
    "--temperature",
    type=float,
    default=0.6,
    help="The temperature is a parameter that controls the randomness of the LLM's output. A higher temperature will result in more creative and imaginative text, while a lower temperature will result in more accurate and factual text.",
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    help="Nucleus sampling threshold",
)
@click.option("--verbose", is_flag=True, default=False)
def main(cache_dir, dataset, dataset_split, max_length, temperature, top_p):
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir


if __name__ == "__main__":
    main()
