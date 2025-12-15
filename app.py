import streamlit as st
from rag_strategies import RAGFactory
from evaluation import RAGEvaluator
from data_loader import DataLoader
from config import Config
import pandas as pd
import json
import os

# Initialize components
rag_factory = RAGFactory()
evaluator = RAGEvaluator()
data_loader = DataLoader()

# Streamlit app
st.set_page_config(layout="wide")
st.title("CritiqueBench - RAG Evaluation Framework")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    rag_strategy = st.selectbox(
        "RAG Strategy",
        ["Simple", "Multi-Query", "HyDE"]
    )
    
    question = st.text_area("Question")
    evaluate_btn = st.button("Evaluate")

# Main content
if evaluate_btn and question:
    # Select RAG strategy
    if rag_strategy == "Simple":
        chain = rag_factory.simple_rag()
    elif rag_strategy == "Multi-Query":
        chain = rag_factory.multi_query_rag()
    else:
        chain = rag_factory.hyde_rag()
    
    # Get response and context
    response = chain.invoke(question)
    context = rag_factory.retriever.invoke(question)
    
    # Evaluate
    evaluation = evaluator.evaluate_response(question, response, context)
    evaluator.save_results(evaluation, f"eval_{rag_strategy.lower()}.json")
    
    # Display results
    st.subheader("Response")
    st.write(response)
    
    st.subheader("Evaluation Metrics")
    
    # LLM-based metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Faithfulness",
            f"{evaluation['faithfulness_score']:.2f}",
            help=evaluation["faithfulness_explanation"]
        )
    with col2:
        st.metric(
            "Answer Relevance",
            f"{evaluation['answer_relevance_score']:.2f}",
            help=evaluation["answer_relevance_explanation"]
        )
    with col3:
        st.metric(
            "Context Relevance",
            f"{evaluation['context_relevance_score']:.2f}",
            help=evaluation["context_relevance_explanation"]
        )
    
    # Traditional metrics
    st.subheader("Traditional NLP Metrics")
    trad_metrics = {
        "BLEU": evaluation["bleu_score"],
        "ROUGE-1 F1": evaluation["rouge1_f1"],
        "ROUGE-2 F1": evaluation["rouge2_f1"],
        "ROUGE-L F1": evaluation["rougeL_f1"]
    }
    st.bar_chart(trad_metrics)
    
    # Context display
    st.subheader("Retrieved Context")
    for doc in context:
        with st.expander(f"Document {context.index(doc)+1}"):
            st.write(doc.page_content)
else:
    st.info("Enter a question and select a RAG strategy to evaluate")

# Initialize data if needed
if not os.path.exists(Config.CHROMA_DIR):
    st.sidebar.info("Initializing vector database...")
    documents = data_loader.load_and_split_documents()
    data_loader.create_vector_store(documents)
    st.sidebar.success("Vector database created!")