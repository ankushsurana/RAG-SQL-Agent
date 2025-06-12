import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()


class LoadToolsConfig:

    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]

        # Internet Search config
        self.tavily_search_max_results = int(
            app_config["tavily_search_api"]["tavily_search_max_results"])

        # Swiss Airline Policy RAG configs
        self.policy_rag_llm = app_config["swiss_airline_policy_rag"]["llm"]
        self.policy_rag_llm_temperature = float(
            app_config["swiss_airline_policy_rag"]["llm_temperature"])
        self.policy_rag_embedding_model = app_config["swiss_airline_policy_rag"]["embedding_model"]
        self.policy_rag_vectordb_directory = str(here(
            app_config["swiss_airline_policy_rag"]["vectordb"])) 
        self.policy_rag_unstructured_docs_directory = str(here(
            app_config["swiss_airline_policy_rag"]["unstructured_docs"]))
        self.policy_rag_k = app_config["swiss_airline_policy_rag"]["k"]
        self.policy_rag_chunk_size = app_config["swiss_airline_policy_rag"]["chunk_size"]
        self.policy_rag_chunk_overlap = app_config["swiss_airline_policy_rag"]["chunk_overlap"]
        self.policy_rag_collection_name = app_config["swiss_airline_policy_rag"]["collection_name"]

        # Novel RAG configs
        self.novel_rag_llm = app_config["novel_rag"]["llm"]
        self.novel_rag_llm_temperature = float(
            app_config["novel_rag"]["llm_temperature"])
        self.novel_rag_embedding_model = app_config["novel_rag"]["embedding_model"]
        self.novel_rag_vectordb_directory = str(here(
            app_config["novel_rag"]["vectordb"]))
        self.novel_rag_unstructured_docs_directory = str(here(
            app_config["novel_rag"]["unstructured_docs"]))
        self.novel_rag_k = app_config["novel_rag"]["k"]
        self.novel_rag_chunk_size = app_config["novel_rag"]["chunk_size"]
        self.novel_rag_chunk_overlap = app_config["novel_rag"]["chunk_overlap"]
        self.novel_rag_collection_name = app_config["novel_rag"]["collection_name"]

        # Travel SQL Agent configs
        self.travel_sqldb_directory = str(here(
            app_config["travel_sqlagent_configs"]["travel_sqldb_dir"]))
        self.travel_sqlagent_llm = app_config["travel_sqlagent_configs"]["llm"]
        self.travel_sqlagent_llm_temperature = float(
            app_config["travel_sqlagent_configs"]["llm_temperature"])

        # Chinook SQL agent configs
        self.chinook_sqldb_directory = str(here(
            app_config["chinook_sqlagent_configs"]["chinook_sqldb_dir"]))
        self.chinook_sqlagent_llm = app_config["chinook_sqlagent_configs"]["llm"]
        self.chinook_sqlagent_llm_temperature = float(
            app_config["chinook_sqlagent_configs"]["llm_temperature"])

        # Graph configs
        self.thread_id = str(
            app_config["graph_configs"]["thread_id"])
