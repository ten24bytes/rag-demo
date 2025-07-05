"""
RAG pipeline implementation using DSPy.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

import dspy
import openai
from loguru import logger

from config import get_settings
from document_processor import DocumentProcessor
from vector_store import VectorStore

settings = get_settings()


class RAGSignature(dspy.Signature):
    """Signature for RAG-based question answering."""
    context = dspy.InputField(desc="Retrieved relevant context from documents")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Comprehensive answer based on the context")


class RAGPipeline(dspy.Module):
    """RAG pipeline using DSPy for optimized retrieval and generation."""

    def __init__(self):
        """Initialize the RAG pipeline."""
        super().__init__()

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()

        # Initialize DSPy language model
        self.lm = self._init_language_model()
        dspy.settings.configure(lm=self.lm)

        # Initialize DSPy modules
        self.generate_answer = dspy.ChainOfThought(RAGSignature)

        logger.info("RAG pipeline initialized successfully")

    def _init_language_model(self) -> dspy.LM:
        """Initialize the language model for DSPy."""
        try:
            # Use dspy.LM directly with OpenAI configuration
            lm = dspy.LM(
                model=f"openai/{settings.llm_model}",
                api_key=settings.openai_api_key,
                max_tokens=1000
            )
            logger.info(f"Language model {settings.llm_model} initialized")
            return lm
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process and add documents to the vector store.

        Args:
            file_paths: List of file paths to process

        Returns:
            Dictionary with processing results
        """
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0,
            "processing_time": ""
        }

        start_time = datetime.now()

        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")

                # Process document and extract text chunks
                chunks = self.document_processor.process_file(file_path)

                if not chunks:
                    results["failed"].append({
                        "file": file_path,
                        "error": "No text content extracted"
                    })
                    continue

                # Generate embeddings
                embeddings = self.document_processor.get_embeddings(chunks)

                if not embeddings:
                    results["failed"].append({
                        "file": file_path,
                        "error": "Failed to generate embeddings"
                    })
                    continue

                # Prepare metadata
                file_name = file_path.split('/')[-1].split('\\')[-1]
                metadata = []
                for i, chunk in enumerate(chunks):
                    metadata.append({
                        "file_name": file_name,
                        "file_path": file_path,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    })

                # Add to vector store
                success = self.vector_store.add_documents(chunks, embeddings, metadata)

                if success:
                    results["success"].append({
                        "file": file_path,
                        "chunks": len(chunks)
                    })
                    results["total_chunks"] += len(chunks)
                else:
                    results["failed"].append({
                        "file": file_path,
                        "error": "Failed to add to vector store - check vector database connection"
                    })

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results["failed"].append({
                    "file": file_path,
                    "error": str(e)
                })

        end_time = datetime.now()
        processing_time = end_time - start_time
        results["processing_time"] = str(processing_time)

        logger.info(f"Document processing completed: {len(results['success'])} successful, "
                    f"{len(results['failed'])} failed")

        return results

    def retrieve_context(self, question: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant context for a question.

        Args:
            question: User's question
            top_k: Number of top chunks to retrieve

        Returns:
            List of relevant text chunks
        """
        try:
            # Generate embedding for the question
            question_embedding = self.document_processor.get_embeddings([question])[0]

            # Search for similar documents
            search_results = self.vector_store.search_similar(
                query_embedding=question_embedding,
                limit=top_k,
                score_threshold=0.1
            )

            # Extract text chunks
            context_chunks = []
            for text, score, metadata in search_results:
                # Add metadata info to the chunk for better context
                chunk_with_meta = f"[From: {metadata.get('file_name', 'Unknown')}]\n{text}"
                context_chunks.append(chunk_with_meta)

            logger.info(f"Retrieved {len(context_chunks)} context chunks for question")
            return context_chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def forward(self, question: str) -> dspy.Prediction:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User's question

        Returns:
            DSPy Prediction with the answer
        """
        try:
            # Retrieve relevant context
            context_chunks = self.retrieve_context(question)

            if not context_chunks:
                context = "No relevant context found in the documents."
            else:
                context = "\n\n".join(context_chunks)

            # Generate answer using DSPy
            prediction = self.generate_answer(context=context, question=question)

            return prediction

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            # Return a fallback prediction
            return dspy.Prediction(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}"
            )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question and return detailed results.

        Args:
            question: User's question

        Returns:
            Dictionary with answer and metadata
        """
        start_time = datetime.now()

        try:
            # Get the prediction
            prediction = self.forward(question)

            # Retrieve context for metadata
            context_chunks = self.retrieve_context(question)

            end_time = datetime.now()
            processing_time = end_time - start_time

            return {
                "question": question,
                "answer": prediction.answer,
                "context_used": len(context_chunks),
                "processing_time": str(processing_time),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "context_used": 0,
                "processing_time": "0:00:00",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of the RAG system.

        Returns:
            Dictionary with system status information
        """
        try:
            # Check vector store health
            vector_store_healthy = self.vector_store.health_check()
            collection_info = self.vector_store.get_collection_info()

            return {
                "vector_store_healthy": vector_store_healthy,
                "collection_info": collection_info,
                "llm_model": settings.llm_model,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "vector_store_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def clear_documents(self) -> bool:
        """
        Clear all documents from the vector store.

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.vector_store.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return False


# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance."""
    global _rag_pipeline

    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()

    return _rag_pipeline
