import torch
from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings(embedding_model: dict) -> HuggingFaceEmbeddings:
    """
    Creates the embedding function from the `embedding_model` section of tools_config.yml. The vector
    databases and the RAG tools both use it, because queries must be embedded the same way as the documents.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_model["model_name"],
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs=embedding_model["encode_kwargs"]
    )
