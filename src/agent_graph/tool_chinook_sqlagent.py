from functools import lru_cache
from typing import List
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.tools import tool
from agent_graph.llm_utils import load_llm
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.sql_utils import create_readonly_engine, extract_sql


TOOLS_CFG = LoadToolsConfig()

MUSIC_TABLES = ["Album", "Artist", "Genre", "MediaType", "Playlist", "PlaylistTrack", "Track"]
BUSINESS_TABLES = ["Customer", "Employee", "Invoice", "InvoiceLine"]


def get_tables(categories: str) -> List[str]:
    """Maps the category names in the LLM output to the corresponding SQL table names.

    Args:
        categories (str): The LLM output, which names the relevant categories.

    Returns:
        List[str]: A list of SQL table names corresponding to the provided categories. If the
        output doesn't name a category, all the tables are returned so that the query can still be written.
    """
    categories = categories.lower()
    tables = []
    if "music" in categories:
        tables.extend(MUSIC_TABLES)
    if "business" in categories:
        tables.extend(BUSINESS_TABLES)
    return tables or MUSIC_TABLES + BUSINESS_TABLES


class ChinookSQLAgent:
    """
    A specialized SQL agent that interacts with the Chinook SQL database using an LLM (Large Language Model).
    """

    def __init__(self, sqldb_directory: str, llm: str, llm_temerature: float) -> None:
        """Initializes the ChinookSQLAgent with the LLM and database connection.
        """
        self.sql_agent_llm = load_llm(repo_id=llm, temperature=llm_temerature)
        self.db = SQLDatabase(create_readonly_engine(sqldb_directory))
        print(self.db.get_usable_table_names())

        category_chain_system = """Return the categories of SQL tables that are relevant to the user question, \
        as a comma-separated list and nothing else. The categories are:

        Music: albums, artists, genres, media types, playlists and tracks
        Business: customers, employees, invoices and sales"""
        category_prompt = ChatPromptTemplate.from_messages(
            [("system", category_chain_system), ("human", "{input}")])
        category_chain = category_prompt | self.sql_agent_llm | StrOutputParser()
        table_chain = category_chain | RunnableLambda(get_tables)

        query_chain = create_sql_query_chain(self.sql_agent_llm, self.db) | RunnableLambda(extract_sql)
        table_chain = {"input": itemgetter("question")} | table_chain

        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=table_chain) | query_chain


@lru_cache(maxsize=1)
def get_agent() -> ChinookSQLAgent:
    return ChinookSQLAgent(
        sqldb_directory=TOOLS_CFG.chinook_sqldb_directory,
        llm=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temerature=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )


@tool
def query_chinook_sqldb(query: str) -> str:
    """Query the Chinook SQL Database of a digital music store, which has its artists, albums, tracks, genres, playlists, customers, employees and invoices. Input should be a search query."""
    agent = get_agent()

    sql_query = agent.full_chain.invoke({"question": query})

    return agent.db.run(sql_query) or "The query returned no rows."
