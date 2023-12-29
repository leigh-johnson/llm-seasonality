import click
import os
from datetime import datetime
import torch
import datasets

from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline

from llm_seasonality.models import InstructEnum, DatasetEnum
from llm_seasonality.prompt import load_prompt


@click.command()
@click.option("--batch-size", type=int, default=1)
@click.option("--cache-dir", default="/mnt/spindle/.tmp")
@click.option(
    "--dataset",
    type=click.Choice(DatasetEnum, case_sensitive=False),
    default=DatasetEnum.GSM8K,
)
@click.option("--dataset-split", type=str, default="test")
@click.option("--dataset-revision", type=str, default="main")
@click.option(
    "--instruct-model",
    type=click.Choice(
        InstructEnum,
        case_sensitive=False,
    ),
    default=InstructEnum.LLAMA2_7B_CHAT_HF,
)
@click.option(
    "--max-length",
    type=int,
    default=4096,
    help="https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens",
)
@click.option(
    "--num-return-sequences",
    type=int,
    default=1,
    help="The number of highest-scoring beams that should be returned when using beam search, see: https://huggingface.co/blog/how-to-generate",
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
def main(
    batch_size,
    cache_dir,
    dataset,
    dataset_split,
    dataset_revision,
    instruct_model,
    max_length,
    num_return_sequences,
    temperature,
    top_p,
    verbose,
):
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    now = int(datetime.now().timestamp())
    dataset_outdir = f"{cache_dir}/experiments/{dataset}_{instruct_model.value}/{now}/"

    tokenizer = AutoTokenizer.from_pretrained(instruct_model.value)

    # generate kwargs are passed to $pipeline_instance.__call__ which is equivalent to $model.generate()
    # These override generation_config.json values: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json
    pipeline_kwargs = dict(
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
    )

    if instruct_model is InstructEnum.LLAMA2_7B_CHAT_HF:
        torch_dtype = torch.bfloat16
        if batch_size > 1:
            # ref: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.padding_side = "left"
            # float16 output is gibberish when input is batched; haven't looked into why yet
    elif instruct_model is InstructEnum.CODELLAMA_7B_INSTRUCT_HF:
        torch_dtype = torch.float16
        pipeline_kwargs["pad_token_id"] = tokenizer.eos_token_id
        if batch_size > 1:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

    # https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaForCausalLM
    # model_kwargs are passed to LlamaForCausalLM.from_pretrained
    # These override config.json values: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
    model_kwargs = dict(temperature=temperature, top_p=top_p, max_length=max_length)

    pipeline = hf_pipeline(
        task="text-generation",
        model=instruct_model.value,
        device_map="auto",
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        model_kwargs=model_kwargs,
        tokenizer=tokenizer,
        num_return_sequences=num_return_sequences,
    )

    prompt_kwargs = dict(
        dataset_revision=dataset_revision,
        dataset_split=dataset_split,
        instruct_model=instruct_model,
    )

    prompt = load_prompt(dataset.value, **prompt_kwargs)
    # dataset = dataset.map()
    # annotate dataset with:
    # 1) formatted prompt (inc date)
    # 2) generated text
    # 3) is answer correct?
    # 4) measure length


if __name__ == "__main__":
    main()
