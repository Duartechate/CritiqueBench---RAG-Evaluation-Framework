import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EVALUATION_MODEL = "gpt-4"
    RAG_MODEL = "gpt-3.5-turbo"
    
    # Vector Store
    CHROMA_DIR = "./chroma_db"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Evaluation
    METRICS = ["faithfulness", "answer_relevance", "context_relevance"]
    TRADITIONAL_METRICS = ["bleu", "rouge"]
    
    # Paths
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"