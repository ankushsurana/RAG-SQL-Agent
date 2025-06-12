import os
from pyprojroot import here


def create_directory(directory_path: str) -> None:
    if not os.path.exists(here(directory_path)):
        os.makedirs(here(directory_path))
