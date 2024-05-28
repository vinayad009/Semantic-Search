import numpy as np
from typing import List, Tuple

# In-memory storage for embeddings and filenames
document_embeddings = []
document_filenames = []

def save_document_embedding(filename: str, embedding: np.ndarray):
    """
    Save the document embedding and its filename.
    """
    document_embeddings.append(embedding)
    document_filenames.append(filename)

def search_documents(query_embedding: np.ndarray) -> List[str]:
    """
    Search for documents that are most similar to the query embedding.
    """
    if not document_embeddings:
        return []

    # Calculate cosine similarity between the query and each document
    similarities = [np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)) for doc_emb in document_embeddings]

    # Get the indices of the top results sorted by similarity
    top_indices = np.argsort(similarities)[::-1]
    
    # Get filenames of the top results
    top_filenames = [document_filenames[idx] for idx in top_indices]
    
    return top_filenames
