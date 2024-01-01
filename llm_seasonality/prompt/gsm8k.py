import evaluate
from llm_seasonality.prompt.base import BasePrompt
from llm_seasonality.models import DatasetEnum

ANSWER_TOKEN = "####"  # indictates the final answer in ground truth


def parse_final_answer(text: str) -> str:
    """
    Parse final result line in GSM8k dataset

    Example input:
    Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72

    Example output:
    72
    """
    return text.split(ANSWER_TOKEN)[-1].strip()


class Gsm8kPrompt(BasePrompt):
    dataset_name: DatasetEnum = DatasetEnum.GSM8K
    task_description: str = "Solve the following middle-school arithmetic problems, using Python code to solve intermediate arithmetic calculations. Wrap code in ``` for readability. Store your result as a variable named 'ans' and print(ans) as the final step."

    def format_example(self):
        question = "If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?"
        example = """```
speed_of_first_train = 60
speed_of_second_train = 30
distance_apart = speed_of_first_train * 3 + speed_of_second_train * 3
ans = distance_apart
print(ans)
```"""
        return f"""{question}
{example}"""

    def format_prompt(self, row):
        question = row["question"]
        examples = self.format_example()
        if self.experiment_dt is not None:
            date_description = f"Today's date is {self.experiment_dt}"
        else:
            date_description = ""
        prompt = f"""<s>[INST] <<SYS>>{date_description}
{self.task_description}


{examples}
<</SYS>>

{question}[/INST]"""
        row[self.col_input] = prompt
        return row

    def calc_accuracy(self, row):
        if row[self.col_error] is False:
            expected = parse_final_answer(row["answer"])
            row[self.col_accuracy] = expected in row[self.col_output]
        else:
            row[self.col_accuracy] = False
        return row

    def calc_codegen_len(self, row) -> str:
        tokenizer = self.pipeline.tokenizer
        token_len = tokenizer.encode(row[self.col_textgen])
        row[self.col_textgen_len] = token_len
        return row

    def calc_perplexity(self, row) -> str:
        perplexity = evaluate.load("perplexity", module_type="metric")
        # calc perplexity of input prompt
        input_perplexity = perplexity.compute(
            batch_size=1,
            model_id=self.instruct_model.value,
            predictions=row[self.col_input],
        )["perplexities"]
        row[self.col_input_perplexity] = input_perplexity

        # calc perplexity of output codegen
        output_perplexity = perplexity.compute(
            batch_size=1,
            model_id=self.instruct_model.value,
            predictions=row[self.col_textgen],
        )["perplexities"]

        row[self.col_output_perplexity] = output_perplexity

        return row


PROMPT = Gsm8kPrompt
