import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import plotly.graph_objects as go
import pandas as pd
import re
from typing import List, Dict
import os
import shutil


if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

def initialize_chroma(embeddings):
    """Initialize ChromaDB vector store"""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def process_pdf(file) -> List[str]:
    """Process PDF and return chunks"""
    
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    
    
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    
    os.remove("temp.pdf")
    return chunks

def extract_financial_metrics(text: str) -> Dict:
    """Extract key financial metrics from text"""
    metrics = {
        'revenue': None,
        'profit': None,
        'eps': None,
    }
    
    
    revenue_pattern = r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?revenue'
    profit_pattern = r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?(?:net\s+)?profit'
    eps_pattern = r'EPS of \$?([\d,]+(?:\.\d+)?)'
    
    
    revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
    if revenue_match:
        metrics['revenue'] = float(revenue_match.group(1).replace(',', ''))
    
    profit_match = re.search(profit_pattern, text, re.IGNORECASE)
    if profit_match:
        metrics['profit'] = float(profit_match.group(1).replace(',', ''))
    
    eps_match = re.search(eps_pattern, text, re.IGNORECASE)
    if eps_match:
        metrics['eps'] = float(eps_match.group(1).replace(',', ''))
    
    return metrics

def create_financial_charts(metrics: Dict):
    """Create financial visualization charts"""
    if any(metrics.values()):
        fig = go.Figure()
        for metric, value in metrics.items():
            if value:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=[metric],
                    y=[value]
                ))
        
        fig.update_layout(
            title="Key Financial Metrics",
            yaxis_title="Value ($)",
            showlegend=False
        )
        return fig
    return None

def main():
    st.title("Financial Report Analysis System")
    
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = initialize_chroma(embeddings)
    
    
    llm = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama2",
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    
    if st.sidebar.button("Reset Database"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR)
            st.session_state.processed_files = []
            st.sidebar.success("Database reset successfully!")
            vectorstore = initialize_chroma(embeddings)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner("Processing document..."):
            # Process the PDF
            chunks = process_pdf(uploaded_file)
            
            
            vectorstore.add_documents(chunks)
            vectorstore.persist()  
            
            
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"Processed {uploaded_file.name}")
            
            # Extract and display metrics
            all_text = " ".join([chunk.page_content for chunk in chunks])
            metrics = extract_financial_metrics(all_text)
            
            # Create and display charts
            fig = create_financial_charts(metrics)
            if fig:
                st.plotly_chart(fig)
    
    # Query interface
    st.subheader("Ask Questions About the Reports")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Generating response..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            response = qa_chain.run(query)
            st.write(response)
    
    # Generate executive summary
    if st.button("Generate Executive Summary"):
        with st.spinner("Generating executive summary..."):
            summary_prompt = """
            Generate an executive summary of the financial report covering:
            1. Key financial highlights
            2. Major developments
            3. Business outlook
            Be concise and focus on the most important information.
            """
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            summary = qa_chain.run(summary_prompt)
            st.subheader("Executive Summary")
            st.write(summary)

if __name__ == "__main__":
    main()