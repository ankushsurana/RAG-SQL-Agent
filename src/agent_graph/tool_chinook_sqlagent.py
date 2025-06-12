from typing import List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.tools import tool
from sqlalchemy import NullPool, create_engine
from agent_graph.load_tools_config import LoadToolsConfig
from langchain_core.runnables import RunnableLambda


TOOLS_CFG = LoadToolsConfig()


class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """

    name: str = Field(description="Name of table in SQL database.")

class TableList(BaseModel):
    """
    Represents a list of tables in the SQL database.

    Attributes:
        tables (List[Table]): A list of relevent Table objects representing the tables in the SQL database.
    """
    tables: List[Table] = Field(description="\nList of relevent tables names.")

def get_tables(categories: TableList) -> List[str]:
    """Maps category names to corresponding SQL table names.

    Args:
        categories (List[Table]): A list of `Table` objects representing different categories.

    Returns:
        List[str]: A list of SQL table names corresponding to the provided categories.
    """
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(
                ["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


class ChinookSQLAgent:
    """
    A specialized SQL agent that interacts with the Chinook SQL database using an LLM (Large Language Model).
    """

    def __init__(self, sqldb_directory: str, llm: str, llm_temerature: float) -> None:
        """Initializes the ChinookSQLAgent with the LLM and database connection.
        """
        self.sql_agent_llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=llm,
                temperature=llm_temerature
            )
        )
        engine = create_engine(
            f"sqlite:///{sqldb_directory}",
            connect_args={"check_same_thread": False},
            poolclass=NullPool
        )
        self.db = SQLDatabase(engine)
        print(self.db.get_usable_table_names())

        category_chain_system = """Return the names of the SQL tables that are relevant to the user question. \
        The tables are:

        Music
        Business"""
        category_chain = create_extraction_chain_pydantic(
            TableList, self.sql_agent_llm, system_message=category_chain_system)
        table_chain = category_chain | RunnableLambda(get_tables)

        query_chain = create_sql_query_chain(self.sql_agent_llm, self.db)
        table_chain = {"input": itemgetter("question")} | table_chain

        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=table_chain) | query_chain


@tool
def query_chinook_sqldb(query: str) -> str:
    """Query the Chinook SQL Database. Input should be a search query."""
    agent = ChinookSQLAgent(
        sqldb_directory=TOOLS_CFG.chinook_sqldb_directory,
        llm=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temerature=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )

    query = agent.full_chain.invoke({"input": query})

    return agent.db.run(query)
