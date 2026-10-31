import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from agent_graph.rag_utils import load_embeddings

class PrepareVectorDB:
    """
    A class to prepare and manage a Vector Database (VectorDB) using documents from a specified directory.
    The class performs the following tasks:
    - Loads and splits documents (PDFs).
    - Splits the text into chunks based on the specified chunk size and overlap.
    - Embeds the document chunks using a specified embedding model.
    - Stores the embedded vectors in a persistent VectorDB directory.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embedding_model: dict,
                 vectordb_dir: str,
                 collection_name: str
                 ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name

    def path_maker(self, file_name: str, doc_dir):
        """
        Creates a full file path by joining the provided directory and file name.
        """
        return os.path.join(here(doc_dir), file_name)

    def run(self):
        """
        Executes the main logic to create and store document embeddings in a VectorDB. An existing
        VectorDB is kept, unless it is empty because an earlier run failed.
        """
        embeddings = load_embeddings(self.embedding_model)
        vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(here(self.vectordb_dir))
        )
        if vectordb._collection.count() > 0:
            print(f"VectorDB '{self.vectordb_dir}' already exists.")
            return

        file_list = [fn for fn in os.listdir(here(self.doc_dir)) if fn.lower().endswith(".pdf")]
        docs = [PyPDFLoader(self.path_maker(
            fn, self.doc_dir)).load() for fn in file_list]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectordb.add_documents(doc_splits)
        print("VectorDB is created and saved.")
        print("Number of vectors in vectordb:",
              vectordb._collection.count(), "\n\n")


if __name__ == "__main__":
    load_dotenv()

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.safe_load(cfg)

    for section in ["swiss_airline_policy_rag", "novel_rag"]:
        prepare_db_instance = PrepareVectorDB(
            doc_dir=app_config[section]["unstructured_docs"],
            chunk_size=app_config[section]["chunk_size"],
            chunk_overlap=app_config[section]["chunk_overlap"],
            embedding_model=app_config[section]["embedding_model"],
            vectordb_dir=app_config[section]["vectordb"],
            collection_name=app_config[section]["collection_name"])

        prepare_db_instance.run()
