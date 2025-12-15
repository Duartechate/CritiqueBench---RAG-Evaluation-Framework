from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from config import Config

class RAGFactory:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=Config.RAG_MODEL, temperature=0)
        self.embedding = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=Config.CHROMA_DIR,
            embedding_function=self.embedding
        )
        self.retriever = self.vectorstore.as_retriever()
        
        # Prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            """
        )
        
    def simple_rag(self):
        """Basic RAG pipeline with simple retrieval"""
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def multi_query_rag(self):
        """RAG with multi-query retrieval"""
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.retriever,
            llm=self.llm
        )
        chain = (
            {"context": multi_retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def hyde_rag(self):
        """RAG with Hypothetical Document Embeddings (HyDE)"""
        hyde_prompt = ChatPromptTemplate.from_template(
            """Please write a hypothetical document that answers this question:
            {question}
            The document should be comprehensive and contain all information needed to answer."""
        )
        
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        
        def hyde_retriever(question):
            hypothetical_doc = hyde_chain.invoke({"question": question})
            return self.retriever.invoke(hypothetical_doc)
        
        chain = (
            {"context": hyde_retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain