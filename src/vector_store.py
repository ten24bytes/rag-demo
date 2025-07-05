"""
Vector store operations using Qdrant database.
"""
import uuid
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from loguru import logger

from config import get_settings

settings = get_settings()


class VectorStore:
    """Manages vector database operations with Qdrant."""

    def __init__(self):
        """Initialize the vector store client."""
        print(f"[VectorStore] Initializing VectorStore...")
        self.client = self._init_client()
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension  # Configurable via environment
        print(f"[VectorStore] About to ensure collection '{self.collection_name}' exists...")
        self._ensure_collection_exists()
        print(f"[VectorStore] VectorStore initialization complete.")

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client."""
        try:
            if settings.qdrant_api_key:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
            else:
                client = QdrantClient(url=settings.qdrant_url)

            # Test connection
            client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {settings.qdrant_url}")
            return client

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant."""
        try:
            print(f"[VectorStore] Checking if collection '{self.collection_name}' exists...")
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            print(f"[VectorStore] Found existing collections: {collection_names}")

            if self.collection_name not in collection_names:
                print(f"[VectorStore] Creating collection: {self.collection_name}")
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"[VectorStore] Collection {self.collection_name} created successfully")
                logger.info(f"Collection {self.collection_name} created successfully")
                
                # Verify the collection was created
                self._verify_collection_exists()
            else:
                print(f"[VectorStore] Collection {self.collection_name} already exists")
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            print(f"[VectorStore] Error ensuring collection exists: {str(e)}")
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def _verify_collection_exists(self):
        """Verify that the collection was successfully created."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection verification successful: {collection_info.config.params.vectors.size} dimensions")
        except Exception as e:
            logger.error(f"Collection verification failed: {str(e)}")
            raise

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries

        Returns:
            True if successful, False otherwise
        """
        try:
            if len(texts) != len(embeddings) != len(metadata):
                raise ValueError("Texts, embeddings, and metadata must have the same length")

            print(f"[VectorStore] add_documents called with {len(texts)} documents")
            logger.info(f"add_documents called with {len(texts)} documents")
            
            # Ensure collection exists before adding documents
            self._ensure_collection_exists_for_operation()

            points = []
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                point_id = str(uuid.uuid4())

                payload = {
                    "text": text,
                    "file_name": meta.get("file_name", "unknown"),
                    "chunk_index": meta.get("chunk_index", i),
                    "file_path": meta.get("file_path", ""),
                    "timestamp": meta.get("timestamp", "")
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Successfully added {len(points)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def _ensure_collection_exists_for_operation(self):
        """Ensure collection exists before performing operations."""
        try:
            print(f"[VectorStore] Checking collection exists for operation...")
            logger.info(f"Checking collection exists for operation...")
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            print(f"[VectorStore] Collection {self.collection_name} confirmed to exist")
            logger.info(f"Collection {self.collection_name} confirmed to exist")
        except Exception as e:
            # Collection doesn't exist, create it
            print(f"[VectorStore] Collection {self.collection_name} not found, creating it now...")
            logger.warning(f"Collection {self.collection_name} not found, creating it now...")
            self._ensure_collection_exists()

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of tuples (text, score, metadata)
        """
        try:
            # Ensure collection exists before searching
            self._ensure_collection_exists_for_operation()
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )

            results = []
            for hit in search_result:
                payload = hit.payload or {}
                text = payload.get("text", "")
                score = hit.score
                metadata = {
                    "file_name": payload.get("file_name", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "file_path": payload.get("file_path", ""),
                    "timestamp": payload.get("timestamp", "")
                }
                results.append((text, score, metadata))

            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def delete_by_file(self, file_name: str) -> bool:
        """
        Delete all documents from a specific file.

        Args:
            file_name: Name of the file to delete documents for

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_name",
                                match=models.MatchValue(value=file_name)
                            )
                        ]
                    )
                )
            )

            logger.info(f"Deleted documents for file: {file_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents for file {file_name}: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            # Ensure collection exists
            self._ensure_collection_exists_for_operation()
            
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": self.embedding_dimension,
                "distance": "COSINE",
                "points_count": info.points_count,
                "status": str(info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                "name": self.collection_name,
                "vector_size": self.embedding_dimension,
                "distance": "COSINE",
                "points_count": 0,
                "status": "ERROR",
                "error": str(e)
            }

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[]
                    )
                )
            )
            logger.info("Collection cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def health_check(self) -> bool:
        """
        Check if the vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False

    def recreate_collection(self) -> bool:
        """
        Recreate the collection (deletes existing one if it exists).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to delete existing collection
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                logger.info(f"Collection {self.collection_name} did not exist")
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Successfully recreated collection: {self.collection_name}")
            
            # Verify the collection was created
            self._verify_collection_exists()
            return True
            
        except Exception as e:
            logger.error(f"Error recreating collection: {str(e)}")
            return False
