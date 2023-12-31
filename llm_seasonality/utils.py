import docker
import re

DOCKER_TAG = "python:3.11"


def extract_python_code(
    row, input_column: str = "generated", output_column="program", regex="```(.*)```"
):
    """
    Extracts python code between separator tokens: ```
    """

    match = re.search(regex, row[input_column], flags=re.DOTALL)

    if match is None:
        row[output_column] = match
    else:
        row[output_column] = match.group(0).replace("```", "")
    return row


def run_python_code(
    row,
    input_column: str = "input",
    output_column="output",
    error_column="error",
):
    command = ["python3", "-c", row[input_column]]
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
