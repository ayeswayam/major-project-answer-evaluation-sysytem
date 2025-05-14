"""
Feedback generation for the Theoretical Answer Evaluation System.
"""

import re
import nltk
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
from preprocessing.text_processor import extract_keywords

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def generate_feedback(student_answer, model_answer, evaluation_result):
    """
    Generate detailed feedback based on evaluation results.
    
    Args:
        student_answer (str): Preprocessed student answer
        model_answer (str): Model answer
        evaluation_result (dict): Dictionary containing evaluation results
    
    Returns:
        str: Detailed feedback for improvement
    """
    # Extract scores
    detailed_scores = evaluation_result["detailed_scores"]
    content_score = detailed_scores["Content Relevance"]
    knowledge_score = detailed_scores["Knowledge Depth"]
    coherence_score = detailed_scores["Coherence"]
    language_score = detailed_scores["Language Quality"]
    keyword_score = detailed_scores["Keyword Coverage"]
    originality_score = detailed_scores["Originality"]
    
    # Generate feedback based on overall score
    overall_score = evaluation_result["overall_score"]
    
    feedback_parts = []
    
    # Opening statement based on overall performance
    if overall_score < 4:
        feedback_parts.append("Your answer needs significant improvement in several areas.")
    elif overall_score < 7:
        feedback_parts.append("Your answer demonstrates basic understanding but requires development in key areas.")
    else:
        feedback_parts.append("Your answer shows good understanding of the topic with some areas for refinement.")
    
    # Content relevance feedback
    if content_score < 0.5:
        feedback_parts.append("The content is not sufficiently relevant to the question. Focus more on directly addressing what was asked.")
    elif content_score < 0.7:
        feedback_parts.append("Your answer is somewhat relevant but could better address the specific aspects of the question.")
    else:
        feedback_parts.append("Good job addressing the question directly and providing relevant content.")
    
    # Knowledge depth feedback
    if knowledge_score < 0.5:
        feedback_parts.append("Your answer lacks depth in key concepts. Try to explain ideas more thoroughly and include more specific details.")
        
        # Suggest missing keywords
        model_keywords = extract_keywords(model_answer, n=5)
        if model_keywords:
            feedback_parts.append(f"Consider including important concepts such as: {', '.join(model_keywords)}.")
    elif knowledge_score < 0.7:
        feedback_parts.append("Your response demonstrates moderate understanding but could benefit from more detailed explanations of core concepts.")
    else:
        feedback_parts.append("Excellent depth of knowledge shown. You've covered key concepts effectively.")
    
    # Coherence feedback
    if coherence_score < 0.5:
        feedback_parts.append("Work on the structure and flow of your answer. Try organizing your ideas into clear paragraphs with appropriate transitions.")
    elif coherence_score < 0.7:
        feedback_parts.append("The organization of your answer is adequate, but could benefit from better paragraph structure and transitions between ideas.")
    else:
        feedback_parts.append("Your answer has good coherence and logical flow.")
    
    # Language quality feedback
    if language_score < 0.5:
        feedback_parts.append("Pay attention to grammar, spelling, and sentence construction. Proofread your work carefully.")
    elif language_score < 0.7:
        feedback_parts.append("Your language is generally clear but watch for minor grammatical or spelling issues.")
    else:
        feedback_parts.append("Your writing demonstrates good language skills and clarity of expression.")
    
    # Keyword coverage feedback
    if keyword_score < 0.5:
        missing_keywords = identify_missing_keywords(student_answer, model_answer)
        feedback_parts.append(f"Your answer is missing important terminology. Try to incorporate key terms such as: {', '.join(missing_keywords[:3])}.")
    elif keyword_score < 0.7:
        feedback_parts.append("You've used some important terminology, but could include more key concepts relevant to this topic.")
    else:
        feedback_parts.append("Good use of subject-specific terminology throughout your answer.")
    
    # Originality feedback
    if originality_score < 0.4:
        feedback_parts.append("Try to express ideas in your own words rather than closely following the standard explanation.")
    elif originality_score > 0.8:
        feedback_parts.append("Your answer shows a unique perspective while covering the essential concepts.")
    
    # Length feedback
    student_length = len(student_answer.split())
    model_length = len(model_answer.split())
    
    if student_length < model_length * 0.5:
        feedback_parts.append("Your answer is quite brief. Consider expanding your explanation with more details and examples.")
    elif student_length > model_length * 1.5:
        feedback_parts.append("Your answer is lengthy. While detail is good, try to be more concise and focused on the key points.")
    
    # Closing remarks and improvement suggestions
    if overall_score < 5:
        feedback_parts.append("To improve, focus on addressing the question more directly and expanding your explanation of key concepts.")
    elif overall_score < 8:
        feedback_parts.append("To enhance your answer, add more specific details and ensure logical flow between your ideas.")
    else:
        feedback_parts.append("To perfect your answer, refine your explanation of more advanced concepts and ensure precise use of terminology.")
    
    # Join all feedback parts
    feedback = " ".join(feedback_parts)
    
    return feedback

def identify_missing_keywords(student_answer, model_answer, n=4):
    """
    Identify important keywords from the model answer that are missing in the student answer.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model answer
        n (int): Number of keywords to return
    
    Returns:
        list: List of missing keywords
    """
    # Extract keywords from model answer
    model_keywords = extract_keywords(model_answer, n=10)
    
    # Check which keywords are missing in student answer
    student_lower = student_answer.lower()
    missing_keywords = []
    
    for keyword in model_keywords:
        if keyword.lower() not in student_lower:
            # Check if any word in the keyword is present
            found = False
            for word in keyword.split():
                if word.lower() in student_lower and len(word) > 3:
                    found = True
                    break
            
            if not found:
                missing_keywords.append(keyword)
    
    return missing_keywords[:n]

def analyze_structure(text):
    """
    Analyze the structure of the answer.
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Structure analysis results
    """
    sentences = sent_tokenize(text)
    paragraphs = text.split('\n\n')
    
    structure = {
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        "has_introduction": False,
        "has_conclusion": False
    }
    
    # Check for introduction and conclusion
    if paragraphs:
        first_para = paragraphs[0].lower()
        if any(word in first_para for word in ["introduction", "initially", "first", "begin", "start"]):
            structure["has_introduction"] = True
            
        last_para = paragraphs[-1].lower()
        if any(word in last_para for word in ["conclusion", "finally", "summary", "overall", "thus", "therefore"]):
            structure["has_conclusion"] = True
    
    return structure

if __name__ == "__main__":
    # Test feedback generation
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
    
    # Mock evaluation result
    evaluation_result = {
        "overall_score": 7.5,
        "detailed_scores": {
            "Content Relevance": 0.85,
            "Knowledge Depth": 0.70,
            "Coherence": 0.65,
            "Language Quality": 0.80,
            "Keyword Coverage": 0.75,
            "Originality": 0.60
        }
    }
    
    feedback = generate_feedback(student_answer, model_answer, evaluation_result)
    print(feedback)
