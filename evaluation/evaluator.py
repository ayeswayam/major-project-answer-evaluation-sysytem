"""
Main evaluation logic for the Theoretical Answer Evaluation System.
"""
import numpy as np
from models.model import get_embeddings, semantic_similarity
from evaluation.metrics import (
    content_relevance_score, 
    knowledge_depth_score,
    coherence_score,
    language_quality_score,
    keyword_coverage_score,
    originality_score
)

def evaluate_answer(student_answer, model_answer, max_score, model):
    """
    Evaluate a student answer against a model answer.
    
    Args:
        student_answer (str): Preprocessed student answer
        model_answer (str): Model answer
        max_score (float): Maximum possible score
        model: Embedding model for semantic analysis
        
    Returns:
        dict: Evaluation results with overall score and detailed metrics
    """
    # Check for empty answers
    if not student_answer.strip():
        return {
            "overall_score": 0,
            "detailed_scores": {
                "Content Relevance": 0,
                "Knowledge Depth": 0,
                "Coherence": 0,
                "Language Quality": 0,
                "Keyword Coverage": 0,
                "Originality": 0
            }
        }
    
    # Get embeddings for semantic comparison
    student_embedding = get_embeddings(model, student_answer)
    model_embedding = get_embeddings(model, model_answer)
    
    # Calculate individual metric scores
    content_score = content_relevance_score(
        student_answer, 
        model_answer, 
        student_embedding, 
        model_embedding
    )
    
    knowledge_score = knowledge_depth_score(
        student_answer, 
        model_answer, 
        student_embedding, 
        model_embedding
    )
    
    coherence_score_value = coherence_score(student_answer)
    
    language_score = language_quality_score(student_answer)
    
    keyword_score = keyword_coverage_score(student_answer, model_answer)
    
    originality_score_value = originality_score(student_answer, model_answer)
    
    # Define weights for each metric (customizable)
    weights = {
        "Content Relevance": 0.30,
        "Knowledge Depth": 0.25,
        "Coherence": 0.15,
        "Language Quality": 0.10,
        "Keyword Coverage": 0.15,
        "Originality": 0.05
    }
    
    # Detailed scores dictionary
    detailed_scores = {
        "Content Relevance": content_score,
        "Knowledge Depth": knowledge_score,
        "Coherence": coherence_score_value,
        "Language Quality": language_score,
        "Keyword Coverage": keyword_score,
        "Originality": originality_score_value
    }
    
    # Calculate weighted average score
    weighted_score = sum(score * weights[metric] for metric, score in detailed_scores.items())
    
    # Scale to max_score
    overall_score = round(weighted_score * max_score, 1)
    
    # Ensure score doesn't exceed max_score
    overall_score = min(overall_score, max_score)
    
    return {
        "overall_score": overall_score,
        "detailed_scores": detailed_scores
    }

def batch_evaluate(student_answers, model_answers, max_scores, model):
    """
    Evaluate multiple student answers in batch.
    
    Args:
        student_answers (list): List of preprocessed student answers
        model_answers (list): List of model answers
        max_scores (list): List of maximum possible scores
        model: Embedding model for semantic analysis
        
    Returns:
        list: List of evaluation results
    """
    results = []
    
    for student_answer, model_answer, max_score in zip(student_answers, model_answers, max_scores):
        result = evaluate_answer(student_answer, model_answer, max_score, model)
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Test evaluation
    from models.model import load_model
    
    # Load model
    model = load_model()
    
    # Sample answers
    student_answer = """
    Inheritance is a key concept in OOP that allows a class to inherit properties and methods from another class.
    The class that inherits is called the subclass and the class that is inherited from is called the superclass.
    This allows for code reuse and establishing relationships between classes.
    """
    
    model_answer = """
    Inheritance is a fundamental concept in object-oriented programming that allows a class to inherit properties
    and methods from another class. The class that inherits is called the subclass or derived class, and the class
    being inherited from is called the superclass or base class. This promotes code reusability and establishes
    a relationship between classes, enabling the creation of hierarchical class structures. Through inheritance,
    subclasses can extend or override functionality of the superclass, implementing the concept of polymorphism.
    """
    
    # Evaluate
    evaluation = evaluate_answer(student_answer, model_answer, 10, model)
    
    # Print results
    print(f"Overall Score: {evaluation['overall_score']}/10")
    print("Detailed Scores:")
    for metric, score in evaluation['detailed_scores'].items():
        print(f"  {metric}: {score:.2f}")
