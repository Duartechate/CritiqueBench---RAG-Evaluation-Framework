from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
import os

class DataLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_and_split_documents(self):
        """Load and split documents from the data directory"""
        loader = DirectoryLoader(
            Config.DATA_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents):
        """Create and persist vector store"""
        from langchain_community.vectorstores import Chroma
        from config import Config
        
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=Config.EMBEDDING_MODEL,
            persist_directory=Config.CHROMA_DIR
        )
        vectordb.persist()
        return vectordb