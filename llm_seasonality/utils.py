import docker
import re
from datasets.formatting.formatting import LazyDict

DOCKER_TAG = "python:3.11"


def extract_python_code(
    row: LazyDict, input_column: str, regex="```(.*)```"
) -> None | str:
    """
    Extracts python code between separator tokens: ```
    """

    match = re.search(regex, row[input_column], flags=re.DOTALL)

    if match is None:
        return None
    else:
        return match.group(0).replace("```", "")


def run_python_code(
    row,
    input_column: str,
    output_column: str,
    error_column: str,
):
    code = extract_python_code(row, input_column)
    command = ["python3", "-c", code]
    client = docker.from_env()
    try:
        result = client.containers.run(
            DOCKER_TAG, command=command, remove=True, stderr=True, stdout=True
        )
        row[output_column] = result.decode("utf8")
        row[error_column] = False
    except docker.errors.ContainerError as e:
        row[output_column] = str(e)
        row[error_column] = True
    return row
