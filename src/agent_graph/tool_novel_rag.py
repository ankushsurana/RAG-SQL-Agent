from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import torch
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class NovelRAGTool:
    """
    A tool for retrieving relevant novel using a Retrieval-Augmented Generation (RAG) approach with vector embeddings.
    """

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str) -> None:
        """
        Initializes the novelRAGTool with the necessary configurations.
        """
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.k = k
        self.vectordb = Chroma(
            collection_name=collection_name,
            persist_directory=self.vectordb_dir,
            embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True})
        )
        print("Number of vectors in vectordb:",
              self.vectordb._collection.count(), "\n\n")


@tool
def lookup_novel(query: str) -> str:
    """Search among the fictional novel and find the answer to the query. Input should be the query."""
    rag_tool = NovelRAGTool(
        embedding_model=TOOLS_CFG.novel_rag_embedding_model,
        vectordb_dir=TOOLS_CFG.novel_rag_vectordb_directory,
        k=TOOLS_CFG.novel_rag_k,
        collection_name=TOOLS_CFG.novel_rag_collection_name)
    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    return "\n\n".join([doc.page_content for doc in docs])
