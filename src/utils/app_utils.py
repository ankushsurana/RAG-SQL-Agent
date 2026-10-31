import os
from pyprojroot import here


def create_directory(directory_path: str) -> None:
    os.makedirs(here(directory_path), exist_ok=True)
