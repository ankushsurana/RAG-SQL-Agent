from functools import lru_cache
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from agent_graph.llm_utils import load_llm
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.sql_utils import create_readonly_engine, extract_sql

TOOLS_CFG = LoadToolsConfig()

class TravelSQLAgentTool:
    """
    A tool for interacting with a travel-related SQL database using an LLM (Language Model) to generate and execute SQL queries.

    This tool enables users to ask travel-related questions, which are transformed into SQL queries by a language model.
    The SQL queries are executed on the provided SQLite database, and the results are processed by the language model to
    generate a final answer for the user.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temerature: float) -> None:
        """
        Initializes the TravelSQLAgentTool with the necessary configurations.
        """
        self.sql_agent_llm = load_llm(repo_id=llm, temperature=llm_temerature)

        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
            Answer:
            """

        self.db = SQLDatabase(create_readonly_engine(sqldb_directory), sample_rows_in_table_info=0)

        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(
            self.sql_agent_llm, self.db) | RunnableLambda(extract_sql)
        answer_prompt = PromptTemplate.from_template(
            self.system_role)

        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        self.chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )


@lru_cache(maxsize=1)
def get_agent() -> TravelSQLAgentTool:
    return TravelSQLAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,
        sqldb_directory=TOOLS_CFG.travel_sqldb_directory,
        llm_temerature=TOOLS_CFG.travel_sqlagent_llm_temperature
    )


@tool
def query_travel_sqldb(query: str) -> str:
    """Query the Swiss Airline SQL Database and access all the company's information, such as flights, bookings, tickets, boarding passes, seats, aircraft and airports. Input should be a search query."""
    agent = get_agent()
    response = agent.chain.invoke({"question": query})
    return response
