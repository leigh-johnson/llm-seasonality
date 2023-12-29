from llm_seasonality.prompt.base import BasePrompt
from llm_seasonality.models import DatasetEnum


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
        prompt = f"""<s>[INST] <<SYS>>
{self.task_description}


{examples}
<</SYS>>

{question}[/INST]"""
        row["prompt"] = prompt
        return row


PROMPT = Gsm8kPrompt
