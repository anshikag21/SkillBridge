import os
import google.generativeai as genai
import numpy as np
from numpy.linalg import norm

# Configure Gemini API using environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_embedding(text: str):
    """Generates text embedding using Gemini API instead of local PyTorch models."""
    if not text:
        return None
    text = text.replace("\n", " ")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="semantic_similarity"
    )
    if result and "embedding" in result:
        return result["embedding"]
    return None

def compute_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts using Gemini embeddings."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    if not emb1 or not emb2:
        return 0.0
        
    vec1, vec2 = np.array(emb1), np.array(emb2)
    norm1, norm2 = norm(vec1), norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(vec1, vec2) / (norm1 * norm2))