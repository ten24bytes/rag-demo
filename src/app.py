"""
Streamlit web application for the RAG demo.
"""
from rag_pipeline import get_rag_pipeline
from document_processor import save_uploaded_file
from config import get_settings, validate_settings
import os
import streamlit as st
from typing import List, Dict, Any
import time
from datetime import datetime

# Configure Loguru logging first
from logging_config import configure_logging
from loguru import logger

# Configure logging when the app starts
configure_logging()

# Import our modules

# Configure page
st.set_page_config(
    page_title="RAG Demo Application",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize settings
settings = get_settings()


def init_session_state():
    """Initialize session state variables."""
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def setup_sidebar():
    """Setup the sidebar with file upload and system controls."""
    st.sidebar.title("🔍 RAG Demo")
    st.sidebar.markdown("---")

    # System status
    st.sidebar.subheader("System Status")

    if st.session_state.rag_pipeline:
        status = st.session_state.rag_pipeline.get_system_status()

        if status.get("vector_store_healthy", False):
            st.sidebar.success("✅ Vector Store: Connected")
        else:
            st.sidebar.error("❌ Vector Store: Disconnected")

        collection_info = status.get("collection_info", {})
        if collection_info:
            st.sidebar.metric("Documents", collection_info.get("points_count", 0))
    else:
        st.sidebar.warning("⚠️ System: Not Initialized")

    st.sidebar.markdown("---")

    # File upload section
    st.sidebar.subheader("📁 Upload Documents")

    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=["txt", "pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Upload text, PDF, DOCX, or PPTX files"
    )

    # Process documents button
    if uploaded_files and st.sidebar.button("🚀 Process Documents", type="primary"):
        process_documents(uploaded_files)

    st.sidebar.markdown("---")

    # Processed files
    st.sidebar.subheader("📚 Processed Files")
    if st.session_state.processed_files:
        for file_info in st.session_state.processed_files:
            st.sidebar.text(f"✓ {file_info['name']} ({file_info['chunks']} chunks)")
    else:
        st.sidebar.text("No files processed yet")

    # Clear documents button
    if st.session_state.processed_files and st.sidebar.button("🗑️ Clear All Documents"):
        clear_documents()

    st.sidebar.markdown("---")

    # Settings
    st.sidebar.subheader("⚙️ Settings")
    st.sidebar.text(f"Embeddings: OpenAI {settings.embedding_model}")
    st.sidebar.text(f"LLM Model: {settings.llm_model}")
    st.sidebar.text(f"Chunk Size: {settings.chunk_size}")


def process_documents(uploaded_files):
    """Process uploaded documents."""
    if not st.session_state.rag_pipeline:
        logger.error("RAG pipeline not initialized when attempting to process documents")
        st.error("RAG pipeline not initialized. Please check your configuration.")
        return

    logger.info(f"Starting to process {len(uploaded_files)} uploaded files")

    with st.spinner("Processing documents..."):
        # Save uploaded files
        file_paths = []
        for uploaded_file in uploaded_files:
            logger.debug(f"Saving uploaded file: {uploaded_file.name}")
            file_path = save_uploaded_file(uploaded_file, settings.upload_dir)
            if file_path:
                file_paths.append(file_path)
                logger.debug(f"Successfully saved file to: {file_path}")
            else:
                logger.warning(f"Failed to save file: {uploaded_file.name}")

        if not file_paths:
            logger.error("No files were successfully saved")
            st.error("Failed to save uploaded files.")
            return

        # Process documents through RAG pipeline
        logger.info(f"Processing {len(file_paths)} files through RAG pipeline")
        results = st.session_state.rag_pipeline.add_documents(file_paths)

        # Update processed files list
        for success_info in results["success"]:
            file_name = success_info["file"].split('/')[-1].split('\\')[-1]
            st.session_state.processed_files.append({
                "name": file_name,
                "chunks": success_info["chunks"],
                "path": success_info["file"]
            })
            logger.info(f"Successfully processed file: {file_name} ({success_info['chunks']} chunks)")

        # Show results
        if results["success"]:
            logger.info(f"Document processing completed: {len(results['success'])} successful, "
                        f"{len(results['failed'])} failed, {results['total_chunks']} total chunks")
            st.success(f"✅ Successfully processed {len(results['success'])} files "
                       f"({results['total_chunks']} chunks)")

        if results["failed"]:
            logger.warning(f"Some documents failed to process: {results['failed']}")
            st.error(f"❌ Failed to process {len(results['failed'])} files")
            for failed in results["failed"]:
                st.error(f"  • {failed['file']}: {failed['error']}")


def clear_documents():
    """Clear all processed documents."""
    if st.session_state.rag_pipeline:
        logger.info("Clearing all processed documents")
        with st.spinner("Clearing documents..."):
            success = st.session_state.rag_pipeline.clear_documents()
            if success:
                logger.info("Successfully cleared all documents from vector store")
                st.session_state.processed_files = []
                st.success("✅ All documents cleared successfully")
            else:
                logger.error("Failed to clear documents from vector store")
                st.error("❌ Failed to clear documents")
    else:
        logger.warning("Attempted to clear documents but RAG pipeline not initialized")


def display_chat_interface():
    """Display the main chat interface."""
    st.title("🤖 RAG Question Answering")

    if not validate_settings():
        st.error("⚠️ Please configure your OpenAI API key in the .env file")
        return

    if not st.session_state.rag_pipeline:
        with st.spinner("Initializing RAG pipeline..."):
            try:
                st.session_state.rag_pipeline = get_rag_pipeline()
                st.success("✅ RAG pipeline initialized successfully")
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG pipeline: {str(e)}")
                return

    # Chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        for i, (question, answer, timestamp) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}{'...' if len(question) > 50 else ''}",
                             expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                st.caption(f"Answered at: {timestamp}")

    # Question input
    st.subheader("❓ Ask a Question")

    if not st.session_state.processed_files:
        st.warning("⚠️ Please upload and process some documents first to ask questions.")
        return

    question = st.text_input(
        "Enter your question:",
        placeholder="Ask anything about your uploaded documents...",
        key="question_input"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        ask_button = st.button("🚀 Ask", type="primary")

    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_button and question:
        answer_question(question)


def answer_question(question: str):
    """Process and answer a user question."""
    with st.spinner("Thinking..."):
        # Get answer from RAG pipeline
        result = st.session_state.rag_pipeline.answer_question(question)

        if result["success"]:
            # Display answer
            st.subheader("💡 Answer")
            st.markdown(result["answer"])

            # Display metadata
            with st.expander("📊 Query Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Context Chunks Used", result["context_used"])
                with col2:
                    st.metric("Processing Time", result["processing_time"])
                with col3:
                    st.text(f"Timestamp: {result['timestamp']}")

            # Add to chat history
            st.session_state.chat_history.append((
                question,
                result["answer"],
                result["timestamp"]
            ))

        else:
            st.error(f"❌ Error: {result['answer']}")


def display_system_info():
    """Display system information and statistics."""
    st.subheader("📊 System Information")

    if st.session_state.rag_pipeline:
        status = st.session_state.rag_pipeline.get_system_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Vector Store Health",
                      "✅ Healthy" if status.get("vector_store_healthy") else "❌ Unhealthy")

        with col2:
            collection_info = status.get("collection_info", {})
            st.metric("Total Documents", collection_info.get("points_count", 0))

        with col3:
            st.metric("Processed Files", len(st.session_state.processed_files))

        # Configuration details
        st.subheader("⚙️ Configuration")
        config_data = {
            "Embeddings": f"OpenAI {settings.embedding_model}",
            "LLM Model": status.get("llm_model", "N/A"),
            "Chunk Size": settings.chunk_size,
            "Chunk Overlap": settings.chunk_overlap,
            "Vector Dimension": collection_info.get("vector_size", "N/A"),
            "Distance Metric": collection_info.get("distance", "N/A")
        }

        for key, value in config_data.items():
            st.text(f"{key}: {value}")


def main():
    """Main application function."""
    init_session_state()
    setup_sidebar()

    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📊 System Info"])

    with tab1:
        display_chat_interface()

    with tab2:
        display_system_info()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Lightweight RAG Demo | Powered by DSPy, Qdrant, and OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
