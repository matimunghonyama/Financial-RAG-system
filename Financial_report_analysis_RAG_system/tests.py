import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  
import plotly.graph_objects as go
import re
from typing import List, Dict
import os
import time

class GemmaFinancialAnalyzer:
    def __init__(self, api_key):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="gemma-7b-it",
            temperature=0.2
        )
        self.vectorstore = None
        self.processing_time = 0

    def process_pdf(self, file):
        start_time = time.time()
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
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.processing_time = time.time() - start_time
        
        return chunks

    def extract_financial_metrics(self, text: str) -> Dict:
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

    def query_documents(self, query):
        start_time = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
        response = qa_chain.run(query)
        query_time = time.time() - start_time
        return response, query_time

def main():
    st.title("Financial Report Analysis - Gemma Model")
    
    
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to use the analysis features.")
        return

    analyzer = GemmaFinancialAnalyzer(groq_api_key)

    
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            chunks = analyzer.process_pdf(uploaded_file)
            
            st.success(f"Processed {uploaded_file.name}")
            st.write(f"Processing Time: {analyzer.processing_time:.2f} seconds")
            
            
            all_text = " ".join([chunk.page_content for chunk in chunks])
            metrics = analyzer.extract_financial_metrics(all_text)
            
            
            if any(metrics.values()):
                st.subheader("Extracted Financial Metrics")
                for metric, value in metrics.items():
                    st.metric(metric.capitalize(), f"${value:,.2f}" if value else "Not Found")

        
        st.subheader("Ask Questions About the Report")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Generating response..."):
                response, query_time = analyzer.query_documents(query)
                st.write(response)
                st.write(f"Query Response Time: {query_time:.2f} seconds")

        
        if st.button("Generate Executive Summary"):
            with st.spinner("Generating executive summary..."):
                summary_prompt = """
                Generate a concise executive summary of the financial report covering:
                1. Key financial highlights
                2. Major developments
                3. Business outlook
                Be brief and focus on the most critical information.
                """
                summary, summary_time = analyzer.query_documents(summary_prompt)
                st.subheader("Executive Summary")
                st.write(summary)

if __name__ == "__main__":
    main()