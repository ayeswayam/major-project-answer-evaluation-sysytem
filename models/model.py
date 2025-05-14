"""
ML model implementation for the Theoretical Answer Evaluation System.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import os

def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a pre-trained language model for text embeddings.

    Args:
        model_name (str): Name of the pre-trained model to load

    Returns:
        SentenceTransformer: Loaded model
    """
    print(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_embeddings(model, text):
    """
    Generate embeddings for text using the provided model.

    Args:
        model: SentenceTransformer model
        text (str): Input text

    Returns:
        numpy.ndarray: Text embedding
    """
    if not text.strip():
        # Return zero vector of appropriate dimension
        return np.zeros(model.get_sentence_embedding_dimension())

    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling - Take attention mask into account for correct averaging.

    Args:
        model_output: Model output
        attention_mask: Attention mask

    Returns:
        torch.Tensor: Mean pooled representation
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings_huggingface(tokenizer, model, text):
    """
    Generate embeddings using HuggingFace models.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        text (str): Input text

    Returns:
        numpy.ndarray: Text embedding
    """
    if not text.strip():
        return np.zeros(768)  # Common default dimension

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embedding.numpy()

def load_huggingface_model(model_name="bert-base-uncased"):
    """
    Load a model and tokenizer from HuggingFace.

    Args:
        model_name (str): Name of the pre-trained model to load

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (numpy.ndarray): First embedding
        embedding2 (numpy.ndarray): Second embedding

    Returns:
        float: Cosine similarity score
    """
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0.0

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_similarity(model, text1, text2):
    """
    Calculate semantic similarity between two texts.

    Args:
        model: Sentence embedding model
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score between 0 and 1
    """
    embedding1 = get_embeddings(model, text1)
    embedding2 = get_embeddings(model, text2)

    return cosine_similarity(embedding1, embedding2)

if __name__ == "__main__":
    # Test the model functions
    model = load_model()
    text1 = "Machine learning is a subset of artificial intelligence."
    text2 = "AI includes various techniques like machine learning."

    emb1 = get_embeddings(model, text1)
    emb2 = get_embeddings(model, text2)

    sim = cosine_similarity(emb1, emb2)
    print(f"Similarity: {sim:.4f}")

    sim2 = semantic_similarity(model, text1, text2)
    print(f"Semantic similarity: {sim2:.4f}")
