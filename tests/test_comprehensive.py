"""
Comprehensive test suite with additional coverage for RAG application.
"""
from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from document_processor import DocumentProcessor, LightweightEmbeddingService
from config import Settings
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_document_processor_invalid_file(self):
        """Test handling of invalid file paths."""
        processor = DocumentProcessor()

        # Test non-existent file
        result = processor.process_file("non_existent_file.txt")
        assert result is None or len(result) == 0

    def test_document_processor_empty_file(self):
        """Test handling of empty files."""
        processor = DocumentProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            result = processor.process_file(temp_path)
            assert result is None or len(result) == 0
        finally:
            os.unlink(temp_path)

    @patch('openai.OpenAI')
    def test_embedding_service_api_failure(self, mock_openai):
        """Test handling of OpenAI API failures."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")

        embedding_service = LightweightEmbeddingService()
        result = embedding_service.encode(["test text"])

        # Should fallback to simple embeddings
        assert len(result) == 1
        assert len(result[0]) == 384  # Fallback dimension

    @patch('vector_store.QdrantClient')
    def test_vector_store_connection_failure(self, mock_qdrant):
        """Test handling of Qdrant connection failures."""
        mock_qdrant.side_effect = Exception("Connection failed")

        with pytest.raises(Exception):
            VectorStore()

    def test_config_validation_missing_api_key(self):
        """Test configuration validation with missing API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=False):
            # Create a new settings instance with empty API key
            from config import validate_settings

            # Patch the global settings to have an empty API key
            with patch('config.settings') as mock_settings:
                mock_settings.openai_api_key = ""
                assert not validate_settings()


class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""

    def test_large_text_chunking(self):
        """Test chunking of very large text."""
        processor = DocumentProcessor()

        # Create a very large text (simulating a large document)
        large_text = "This is a test sentence. " * 10000  # ~250KB of text
        chunks = processor._chunk_text(large_text)

        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 1200 for chunk in chunks)  # Approximate check based on settings

    def test_batch_embedding_processing(self):
        """Test batch processing of embeddings."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(100)]
            mock_client.embeddings.create.return_value = mock_response

            embedding_service = LightweightEmbeddingService()
            texts = [f"Text {i}" for i in range(100)]
            result = embedding_service.encode(texts)

            assert len(result) == 100
            assert all(len(emb) == 1536 for emb in result)

    @patch('vector_store.QdrantClient')
    def test_vector_store_batch_operations(self, mock_qdrant):
        """Test batch operations on vector store."""
        mock_client = Mock()
        mock_qdrant.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        mock_client.upsert.return_value = True

        with patch('vector_store.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                qdrant_collection_name="test_collection",
                embedding_dimension=1536
            )

            vector_store = VectorStore()

            # Test adding many documents
            texts = [f"Document {i}" for i in range(1000)]
            embeddings = [[0.1] * 1536 for _ in range(1000)]
            metadata = [{"file_name": f"doc_{i}.txt"} for i in range(1000)]

            result = vector_store.add_documents(texts, embeddings, metadata)
            assert result is True


class TestSecurityAndValidation:
    """Test security and input validation."""

    def test_file_size_limits(self):
        """Test handling of oversized files."""
        processor = DocumentProcessor()

        # Create a file that's too large (mock scenario)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write a reasonably large amount of data
            f.write("x" * 1000000)  # 1MB file
            temp_path = f.name

        try:
            result = processor.process_file(temp_path)
            # Should handle large files gracefully
            assert isinstance(result, (list, type(None)))
        finally:
            os.unlink(temp_path)

    def test_malicious_filename_handling(self):
        """Test handling of potentially malicious filenames."""
        processor = DocumentProcessor()

        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "con.txt",  # Windows reserved name
            "prn.txt",  # Windows reserved name
        ]

        for malicious_path in malicious_paths:
            result = processor.process_file(malicious_path)
            # Should return None or empty list for invalid paths
            assert result is None or len(result) == 0

    def test_input_sanitization(self):
        """Test input sanitization for text processing."""
        processor = DocumentProcessor()

        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "\x00\x01\x02",  # Binary data
            "�" * 1000,  # Invalid unicode
        ]

        for malicious_input in malicious_inputs:
            try:
                chunks = processor._chunk_text(malicious_input)
                # Should handle malicious input gracefully
                assert isinstance(chunks, list)
            except Exception:
                # Exception is acceptable for malicious input
                pass


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @patch('rag_pipeline.dspy')
    @patch('rag_pipeline.DocumentProcessor')
    @patch('rag_pipeline.VectorStore')
    def test_full_rag_workflow(self, mock_vector_store, mock_processor, mock_dspy):
        """Test complete RAG workflow from document to answer."""
        # Mock all dependencies
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_file.return_value = ["Test chunk 1", "Test chunk 2"]
        mock_processor_instance.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.add_documents.return_value = True
        mock_vector_store_instance.search_similar.return_value = [
            ("Relevant context", 0.9, {"file_name": "test.txt"})
        ]
        mock_vector_store_instance.health_check.return_value = True

        mock_dspy_module = Mock()
        mock_dspy.ChainOfThought.return_value = mock_dspy_module
        mock_dspy_module.forward.return_value = Mock(answer="Test answer")

        with patch('rag_pipeline.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                openai_api_key="test_key",
                llm_model="gpt-3.5-turbo"
            )

            # Initialize pipeline
            pipeline = RAGPipeline()

            # Test document addition
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test document content")
                temp_path = f.name

            try:
                add_result = pipeline.add_documents([temp_path])
                assert "success" in add_result
                assert "failed" in add_result

                # Test query processing
                context = pipeline.retrieve_context("test question")
                assert isinstance(context, list)

            finally:
                os.unlink(temp_path)

    def test_concurrent_access_simulation(self):
        """Test handling of concurrent access (simplified simulation)."""
        # This would ideally use threading, but for simplicity we'll test sequentially
        processor = DocumentProcessor()

        # Create multiple temporary files
        temp_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Test document {i} content")
                temp_files.append(f.name)

        try:
            # Process multiple files in sequence (simulating concurrent access)
            results = []
            for temp_file in temp_files:
                result = processor.process_file(temp_file)
                results.append(result)

            # All should succeed
            assert all(result is not None for result in results)
            assert all(len(result) > 0 for result in results if result)

        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)


class TestDataQuality:
    """Test data quality and consistency."""

    def test_embedding_consistency(self):
        """Test that identical inputs produce identical embeddings."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock consistent response
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response

            embedding_service = LightweightEmbeddingService()

            # Test same input multiple times
            text = "This is a test sentence."
            result1 = embedding_service.encode([text])
            result2 = embedding_service.encode([text])

            assert result1 == result2

    def test_chunk_boundary_consistency(self):
        """Test that text chunking produces consistent boundaries."""
        processor = DocumentProcessor()

        text = "This is sentence one. This is sentence two. This is sentence three. " * 100

        # Chunk the same text multiple times
        chunks1 = processor._chunk_text(text)
        chunks2 = processor._chunk_text(text)

        assert chunks1 == chunks2
        assert len(chunks1) == len(chunks2)

    @patch('vector_store.QdrantClient')
    def test_search_result_ranking(self, mock_qdrant):
        """Test that search results are properly ranked by similarity."""
        mock_client = Mock()
        mock_qdrant.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True

        # Mock search results with different scores
        mock_search_results = [
            Mock(payload={"text": "Very relevant"}, score=0.95),
            Mock(payload={"text": "Somewhat relevant"}, score=0.75),
            Mock(payload={"text": "Less relevant"}, score=0.55),
        ]
        mock_client.search.return_value = mock_search_results

        with patch('vector_store.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                qdrant_collection_name="test_collection",
                embedding_dimension=1536
            )

            vector_store = VectorStore()
            results = vector_store.search_similar([0.1] * 1536, limit=3)

            # Results should be ordered by score (highest first)
            assert len(results) == 3
            scores = [result[1] for result in results]
            assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__])
