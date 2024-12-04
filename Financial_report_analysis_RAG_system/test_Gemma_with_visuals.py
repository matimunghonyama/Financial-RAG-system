import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
from typing import List, Dict
import os
import time
import numpy as np

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
        self.financial_data = None

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
            'assets': None,
            'liabilities': None,
            'equity': None
        }
        
        patterns = {
            'revenue': r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?revenue',
            'profit': r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?(?:net\s+)?profit',
            'eps': r'EPS of \$?([\d,]+(?:\.\d+)?)',
            'assets': r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?total\s+assets',
            'liabilities': r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?total\s+liabilities',
            'equity': r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:in\s+)?shareholders?\s+equity'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics[metric] = float(match.group(1).replace(',', ''))
        
        self.financial_data = metrics
        return metrics

    def generate_financial_plots(self):
        if not self.financial_data:
            st.warning("No financial data available for plotting.")
            return None

        # Remove None values
        plot_data = {k: v for k, v in self.financial_data.items() if v is not None}
        
        # Create multiple visualizations
        plots = []

        # Bar Chart of Financial Metrics
        if plot_data:
            fig1 = go.Figure(data=[
                go.Bar(
                    x=list(plot_data.keys()),
                    y=list(plot_data.values()),
                    text=[f'${v:,.2f}' for v in plot_data.values()],
                    textposition='auto'
                )
            ])
            fig1.update_layout(
                title='Financial Metrics Overview',
                xaxis_title='Metric',
                yaxis_title='Value ($)',
                height=400
            )
            plots.append(('Bar Chart of Financial Metrics', fig1))

        # Pie Chart of Financial Composition
        if len(plot_data) > 2:
            fig2 = go.Figure(data=[
                go.Pie(
                    labels=list(plot_data.keys()),
                    values=list(plot_data.values()),
                    hole=.3
                )
            ])
            fig2.update_layout(
                title='Financial Composition',
                height=400
            )
            plots.append(('Financial Composition Pie Chart', fig2))

        # Radar Chart of Financial Metrics
        if len(plot_data) > 2:
            fig3 = go.Figure(data=go.Scatterpolar(
                r=list(plot_data.values()),
                theta=list(plot_data.keys()),
                fill='toself'
            ))
            fig3.update_layout(
                title='Radar Chart of Financial Metrics',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(plot_data.values()) * 1.1]
                    )
                ),
                height=400
            )
            plots.append(('Radar Chart of Financial Metrics', fig3))

        return plots

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
    
    # Groq API Configuration
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to use the analysis features.")
        return

    analyzer = GemmaFinancialAnalyzer(groq_api_key)

    # File upload
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            chunks = analyzer.process_pdf(uploaded_file)
            
            st.success(f"Processed {uploaded_file.name}")
            st.write(f"Processing Time: {analyzer.processing_time:.2f} seconds")
            
            # Extract and display metrics
            all_text = " ".join([chunk.page_content for chunk in chunks])
            metrics = analyzer.extract_financial_metrics(all_text)
            
            # Metrics display
            if any(metrics.values()):
                st.subheader("Extracted Financial Metrics")
                for metric, value in metrics.items():
                    st.metric(metric.capitalize(), f"${value:,.2f}" if value else "Not Found")

                # Generate and display plots
                plots = analyzer.generate_financial_plots()
                if plots:
                    st.subheader("Financial Data Visualizations")
                    for title, fig in plots:
                        st.plotly_chart(fig)

        # Query interface
        st.subheader("Ask Questions About the Report")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Generating response..."):
                response, query_time = analyzer.query_documents(query)
                st.write(response)
                st.write(f"Query Response Time: {query_time:.2f} seconds")

        # Executive Summary
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