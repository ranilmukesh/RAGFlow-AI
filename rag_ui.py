import streamlit as st
import asyncio
from pathlib import Path
import pandas as pd
import plotly.express as px
from typing import List, Dict
import time
from datetime import datetime
import multiprocessing
import sys
from imports import *
from processor import EnhancedDocumentProcessor
from rag_system import RAGSystem

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class RAGUI:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize_session_state()
        self.setup_ui_config()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'mode' not in st.session_state:
            st.session_state.mode = None  # 'sources', 'upload', or 'chat'
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'current_documents' not in st.session_state:
            st.session_state.current_documents = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
    def setup_ui_config(self):
        """Setup UI configuration with API keys and URLs"""
        st.sidebar.title("Configuration")
        
        with st.sidebar.expander("API Keys", expanded=False):
            self.config["groq_api_key"] = st.text_input("Groq API Key", type="password")
            self.config["claude_api_key"] = st.text_input("Claude API Key", type="password")
            self.config["openai_api_key"] = st.text_input("OpenAI API Key", type="password")
        
        with st.sidebar.expander("SharePoint Config", expanded=False):
            self.config["sharepoint_url"] = st.text_input("SharePoint URL")
            self.config["sharepoint_client_id"] = st.text_input("Client ID")
            self.config["sharepoint_client_secret"] = st.text_input("Client Secret", type="password")
        
        with st.sidebar.expander("Salesforce Config", expanded=False):
            self.config["sf_username"] = st.text_input("Salesforce Username")
            self.config["sf_password"] = st.text_input("Salesforce Password", type="password")
            self.config["sf_token"] = st.text_input("Security Token", type="password")
        
        # Save configuration button
        if st.sidebar.button("Save Configuration"):
            st.session_state.config = self.config
            st.success("Configuration saved!")
            
        # Initialize RAG system only after configuration is set
        if all(self.config.get(key) for key in ["groq_api_key", "claude_api_key", "openai_api_key"]):
            if "rag_system" not in st.session_state:
                st.session_state.rag_system = RAGSystem(self.config)

    def authenticate(self):
        """Simple authentication system"""
        st.sidebar.header("Authentication")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Replace with secure check
                st.session_state.authenticated = True
                st.sidebar.success("Authenticated successfully!")
            else:
                st.sidebar.error("Invalid credentials")

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("Configuration")
            
            # API Selection
            st.subheader("API Selection")
            api_choice = st.radio(
                "Choose LLM API",
                options=["Claude", "OpenAI", "Groq"]
            )
            
            # Document Sources
            st.subheader("Document Sources")
            sources = {
                "SharePoint": st.checkbox("SharePoint", value=True),
                "Salesforce": st.checkbox("Salesforce"),
                "Local Files": st.checkbox("Local Files", value=True)
            }
            
            # Document Types
            st.subheader("Document Types")
            doc_types = {
                ".pdf": st.checkbox("PDF Files", value=True),
                ".docx": st.checkbox("Word Documents", value=True),
                ".pptx": st.checkbox("PowerPoint Presentations"),
                ".xlsx": st.checkbox("Excel Spreadsheets"),
                ".html": st.checkbox("HTML Files"),
                ".eml": st.checkbox("Email Messages"),
                ".msg": st.checkbox("Outlook Messages"),
                ".txt": st.checkbox("Text Files")
            }
            
            # Advanced Settings
            with st.expander("Advanced Settings"):
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
                
                max_results = st.number_input(
                    "Max Results",
                    min_value=1,
                    max_value=20,
                    value=5
                )
            
            return api_choice, sources, doc_types, confidence_threshold, max_results

    async def process_documents(self, files, sources, doc_types):
        """Process uploaded documents"""
        allowed_types = [ext for ext, enabled in doc_types.items() if enabled]
        allowed_sources = [source for source, enabled in sources.items() if enabled]
        
        try:
            # Initialize RAG system if not already done
            if not st.session_state.rag_system:
                config = self.load_config()
                st.session_state.rag_system = RAGSystem(config)
            
            documents = []
            for file in files:
                if Path(file.name).suffix in allowed_types:
                    # Save uploaded file temporarily
                    temp_path = Path(f"temp/{file.name}")
                    temp_path.parent.mkdir(exist_ok=True)
                    temp_path.write_bytes(file.getvalue())
                    documents.append(str(temp_path))
            
            return documents
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return []

    async def query_documents(self, query: str, confidence_threshold: float, api_choice: str):
        """Query processed documents"""
        try:
            if "rag_system" not in st.session_state:
                st.error("Please configure API keys first")
                return None
            
            result = await st.session_state.rag_system.query(
                query=query,
                min_confidence=confidence_threshold,
                user_id=st.session_state.get('user_id')
            )
            
            # Process with selected LLM
            llm_result = await st.session_state.rag_system.processor.process_with_llm(
                content=f"Context: {result['context']}\nQuery: {query}",
                llm_name=api_choice
            )
            
            return {
                **result,
                "llm_response": llm_result["text"],
                "model_used": llm_result["model"],
                "provider": llm_result["provider"]
            }
            
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def render_results(self, result: Dict):
        """Render query results"""
        if not result:
            return
            
        # Answer Section
        st.header("Answer")
        st.markdown(f"**{result['llm_response']}**")
        
        # Sources Section
        st.subheader("Sources")
        sources_df = pd.DataFrame({
            'Source': result['sources'],
            'Confidence': result['confidence_scores']
        })
        
        # Display sources as a table and chart
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(sources_df)
        with col2:
            fig = px.bar(sources_df, x='Source', y='Confidence',
                        title='Source Confidence Scores')
            st.plotly_chart(fig)
            
        # Metadata Section
        if result.get('metadata'):
            with st.expander("Document Metadata"):
                st.json(result['metadata'])
        
        # Export Results
        self.export_results(result)

    def export_results(self, result: Dict):
        """Export query results to CSV"""
        if st.button("Export Results"):
            df = pd.DataFrame({
                'Source': result['sources'],
                'Confidence': result['confidence_scores']
            })
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='query_results.csv',
                mime='text/csv'
            )

    def render_analytics(self):
        """Render analytics dashboard"""
        if not st.session_state.processing_history:
            return
            
        st.header("Analytics")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_queries = len(st.session_state.processing_history)
            st.metric("Total Queries", total_queries)
            
        with col2:
            avg_confidence = np.mean([
                np.mean(h['result']['confidence_scores'])
                for h in st.session_state.processing_history
            ])
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
        with col3:
            total_docs = len(st.session_state.current_documents)
            st.metric("Documents Processed", total_docs)
            
        # Query History
        st.subheader("Query History")
        history_df = pd.DataFrame([
            {
                'Timestamp': pd.to_datetime(h['timestamp'], unit='s'),
                'Query': h['query'],
                'Avg Confidence': np.mean(h['result']['confidence_scores'])
            }
            for h in st.session_state.processing_history
        ])
        
        fig = px.line(history_df, x='Timestamp', y='Avg Confidence',
                     title='Confidence Scores Over Time')
        st.plotly_chart(fig)

    def render_document_preview(self, file):
        """Render document preview"""
        with st.expander(f"Preview: {file.name}"):
            if file.type == "application/pdf":
                st.write("PDF preview not supported in this demo.")
            elif "text" in file.type:
                st.code(file.getvalue().decode())

    def main(self):
        """Main UI rendering function"""
        st.title("🤖 RAG System Interface")
        
        # Mode Selection
        if not st.session_state.mode:
            st.write("### Choose Your Interaction Mode")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📁 Connect to Data Sources", use_container_width=True):
                    st.session_state.mode = 'sources'
                    st.rerun()
            
            with col2:
                if st.button("📤 Upload Documents", use_container_width=True):
                    st.session_state.mode = 'upload'
                    st.rerun()
            
            with col3:
                if st.button("���� Start Chatting", use_container_width=True):
                    st.session_state.mode = 'chat'
                    st.rerun()
                    
        else:
            # Show mode switcher in sidebar
            st.sidebar.title("Mode")
            new_mode = st.sidebar.radio(
                "Select Mode",
                ['sources', 'upload', 'chat'],
                format_func=lambda x: {
                    'sources': '📁 Data Sources',
                    'upload': '📤 Upload',
                    'chat': '💬 Chat'
                }[x],
                index=['sources', 'upload', 'chat'].index(st.session_state.mode)
            )
            
            if new_mode != st.session_state.mode:
                st.session_state.mode = new_mode
                st.rerun()
            
            # Render appropriate interface based on mode
            if st.session_state.mode == 'sources':
                self.render_sources_interface()
            elif st.session_state.mode == 'upload':
                self.render_upload_interface()
            else:  # chat mode
                self.render_chat_interface()
                
    def render_sources_interface(self):
        """Render data sources connection interface"""
        st.header("Connect to Data Sources")
        
        # SharePoint Connection
        with st.expander("SharePoint", expanded=True):
            sharepoint_url = st.text_input("SharePoint URL")
            sharepoint_user = st.text_input("Username")
            sharepoint_pass = st.text_input("Password", type="password")
            if st.button("Connect to SharePoint"):
                with st.spinner("Connecting to SharePoint..."):
                    try:
                        self.processor.integration_manager.connect_sharepoint(
                            url=sharepoint_url,
                            username=sharepoint_user,
                            password=sharepoint_pass
                        )
                        st.success("Connected to SharePoint!")
                    except Exception as e:
                        st.error(f"Failed to connect: {str(e)}")
        
        # Salesforce Connection
        with st.expander("Salesforce", expanded=True):
            sf_username = st.text_input("Salesforce Username")
            sf_password = st.text_input("Salesforce Password", type="password")
            sf_token = st.text_input("Security Token")
            if st.button("Connect to Salesforce"):
                with st.spinner("Connecting to Salesforce..."):
                    try:
                        self.processor.integration_manager.connect_salesforce(
                            username=sf_username,
                            password=sf_password,
                            token=sf_token
                        )
                        st.success("Connected to Salesforce!")
                    except Exception as e:
                        st.error(f"Failed to connect: {str(e)}")
                    
    def render_upload_interface(self):
        """Render document upload interface"""
        st.header("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx']
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    total_files = len(uploaded_files)
                    
                    for i, file in enumerate(uploaded_files):
                        # Simulate processing
                        st.session_state.current_documents.append(file.name)
                        progress_bar.progress((i + 1) / total_files)
                        st.write(f"Processed {file.name}")
                    
                    st.success("Documents processed successfully!")
                    
    def render_chat_interface(self):
        """Render chat interface"""
        st.header("Chat with Your Documents")
        
        # Get configuration from sidebar
        api_choice, _, _, confidence_threshold, _ = self.render_sidebar()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Process query
                    result = asyncio.run(self.query_documents(
                        prompt, 
                        confidence_threshold,
                        api_choice
                    ))
                    
                    if result:
                        st.write(result["llm_response"])
                        
                        # Show context if available
                        if "context" in result:
                            with st.expander("View Source Context"):
                                st.info(result["context"])
                        
                        # Show sources if available
                        if "sources" in result:
                            with st.expander("View Sources"):
                                for source, confidence in zip(result["sources"], result["confidence_scores"]):
                                    st.write(f"- {source} ({confidence:.0%} confidence)")
                    
                    # Add assistant message to chat history
                    if result and "llm_response" in result:
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": result["llm_response"]
                        })