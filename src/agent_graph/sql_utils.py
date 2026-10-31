import re
import sqlite3
from pathlib import Path
from sqlalchemy import Engine, NullPool, create_engine


def create_readonly_engine(sqldb_directory: str) -> Engine:
    """
    Creates an engine for a SQLite database that is opened read-only, so that SQL written by the LLM
    can't change the data and a missing database file raises an error instead of being created empty.
    """
    uri = f"{Path(sqldb_directory).as_uri()}?mode=ro"
    return create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False),
        poolclass=NullPool
    )


def extract_sql(text: str) -> str:
    """
    Extracts the SQL statement from the LLM output, which can include an explanation, a "SQLQuery:"
    prefix, a markdown code block, or a made-up "SQLResult:" section after the query.
    """
    code_block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    prefix = re.search(r"SQLQuery:\s*", text, re.IGNORECASE)
    if prefix:
        text = text[prefix.end():]
    text = re.split(r"\b(?:SQLResult|Answer):", text, flags=re.IGNORECASE)[0]

    statement = re.search(r"\b(?:SELECT|WITH)\b.*?(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    return statement.group(0).strip() if statement else text.strip()
