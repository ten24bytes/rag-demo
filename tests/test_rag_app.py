"""
Test suite for the RAG application.
"""
from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config import Settings, get_settings
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfig:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.embedding_model == "text-embedding-3-small"  # Updated to match actual default

    def test_settings_from_env(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {'CHUNK_SIZE': '500', 'LLM_MODEL': 'gpt-4'}):
            settings = Settings()
            assert settings.chunk_size == 500
            assert settings.llm_model == 'gpt-4'


class TestDocumentProcessor:
    """Test document processing functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.processor = DocumentProcessor()

    def test_text_chunking(self):
        """Test text chunking functionality."""
        text = "This is a test sentence. " * 100  # Long text
        chunks = self.processor._chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_supported_formats(self):
        """Test supported file formats."""
        expected_formats = {'.txt', '.pdf', '.docx', '.pptx'}
        actual_formats = set(self.processor.supported_formats.keys())
        assert actual_formats == expected_formats

    def test_process_text_file(self):
        """Test processing a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document with multiple sentences. It contains various information.")
            temp_path = f.name

        try:
            chunks = self.processor.process_file(temp_path)
            assert chunks is not None
            assert len(chunks) > 0
            assert isinstance(chunks[0], str)
        finally:
            os.unlink(temp_path)

    def test_get_embeddings(self):
        """Test embedding generation."""
        texts = ["This is a test sentence.", "This is another test sentence."]
        embeddings = self.processor.get_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)


class TestVectorStore:
    """Test vector store functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        with patch('vector_store.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock collection methods
            mock_instance.get_collections.return_value = Mock(collections=[])
            mock_instance.create_collection.return_value = True
            mock_instance.upsert.return_value = True
            mock_instance.search.return_value = []

            yield mock_instance

    def test_vector_store_initialization(self, mock_qdrant_client):
        """Test vector store initialization."""
        with patch('vector_store.settings') as mock_settings:
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.qdrant_api_key = None
            mock_settings.qdrant_collection_name = "test_collection"
            mock_settings.embedding_dimension = 1536

            vector_store = VectorStore()
            assert vector_store.client is not None
            assert vector_store.collection_name == "test_collection"

    def test_add_documents(self, mock_qdrant_client):
        """Test adding documents to vector store."""
        with patch('vector_store.settings') as mock_settings:
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.qdrant_api_key = None
            mock_settings.qdrant_collection_name = "test_collection"
            mock_settings.embedding_dimension = 1536

            vector_store = VectorStore()

            texts = ["Test document 1", "Test document 2"]
            embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            metadata = [{"file_name": "test1.txt"}, {"file_name": "test2.txt"}]

            result = vector_store.add_documents(texts, embeddings, metadata)
            assert result is True


class TestRAGPipeline:
    """Test RAG pipeline functionality."""

    @pytest.fixture
    def mock_components(self):
        """Mock all RAG pipeline components."""
        with patch('rag_pipeline.DocumentProcessor') as mock_processor, \
                patch('rag_pipeline.VectorStore') as mock_vector_store, \
                patch('rag_pipeline.dspy') as mock_dspy:

            # Mock document processor
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance
            mock_processor_instance.process_file.return_value = ["Test chunk 1", "Test chunk 2"]
            mock_processor_instance.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

            # Mock vector store
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.add_documents.return_value = True
            mock_vector_store_instance.search_similar.return_value = [
                ("Test context", 0.9, {"file_name": "test.txt"})
            ]
            mock_vector_store_instance.health_check.return_value = True

            # Mock DSPy
            mock_dspy_instance = Mock()
            mock_dspy.return_value = mock_dspy_instance

            yield {
                "processor": mock_processor_instance,
                "vector_store": mock_vector_store_instance,
                "dspy": mock_dspy_instance
            }

    def test_rag_pipeline_initialization(self, mock_components):
        """Test RAG pipeline initialization."""
        with patch('rag_pipeline.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                openai_api_key="test_key",
                llm_model="gpt-3.5-turbo"
            )

            pipeline = RAGPipeline()
            assert pipeline.document_processor is not None
            assert pipeline.vector_store is not None

    def test_add_documents(self, mock_components):
        """Test adding documents to the pipeline."""
        with patch('rag_pipeline.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                openai_api_key="test_key",
                llm_model="gpt-3.5-turbo"
            )

            pipeline = RAGPipeline()

            # Create temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test document content")
                temp_path = f.name

            try:
                results = pipeline.add_documents([temp_path])
                assert "success" in results
                assert "failed" in results
                assert "total_chunks" in results
            finally:
                os.unlink(temp_path)

    def test_retrieve_context(self, mock_components):
        """Test context retrieval."""
        with patch('rag_pipeline.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                openai_api_key="test_key",
                llm_model="gpt-3.5-turbo"
            )

            pipeline = RAGPipeline()
            context = pipeline.retrieve_context("test question")

            assert isinstance(context, list)


def test_integration():
    """Test basic integration between components."""
    # This is a simple integration test that doesn't require external services
    with patch('config.validate_settings') as mock_validate:
        mock_validate.return_value = True

        settings = get_settings()
        assert settings is not None


if __name__ == "__main__":
    pytest.main([__file__])
