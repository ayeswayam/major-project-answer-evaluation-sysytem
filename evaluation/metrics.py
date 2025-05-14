"""
Evaluation metrics for the Theoretical Answer Evaluation System.
"""
import re
import nltk
import numpy as np
import textstat
import spacy
import jellyfish
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from rouge import Rouge
from preprocessing.text_processor import extract_keywords

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def content_relevance_score(student_answer, model_answer, student_embedding, model_embedding):
    """
    Calculate content relevance based on semantic similarity.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model answer
        student_embedding (numpy.ndarray): Student answer embedding
        model_embedding (numpy.ndarray): Model answer embedding
        
    Returns:
        float: Content relevance score (0-1)
    """
    # Semantic similarity using embeddings
    if np.all(student_embedding == 0) or np.all(model_embedding == 0):
        return 0.0
    
    dot_product = np.dot(student_embedding, model_embedding)
    norm1 = np.linalg.norm(student_embedding)
    norm2 = np.linalg.norm(model_embedding)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Add ROUGE score for lexical similarity
    try:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(student_answer, model_answer)[0]
        rouge_l_f = rouge_scores['rouge-l']['f']
    except Exception:
        # Handle case where ROUGE fails (e.g., very short answers)
        rouge_l_f = 0.0
    
    # Combine semantic and lexical similarity
    combined_score = 0.7 * cosine_sim + 0.3 * rouge_l_f
    
    return min(max(combined_score, 0.0), 1.0)

def knowledge_depth_score(student_answer, model_answer, student_embedding, model_embedding):
    """
    Assess the depth of knowledge demonstrated in the answer.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model answer
        student_embedding (numpy.ndarray): Student answer embedding
        model_embedding (numpy.ndarray): Model answer embedding
        
    Returns:
        float: Knowledge depth score (0-1)
    """
    # Extract keywords from model answer
    model_keywords = extract_keywords(model_answer, n=15)
    
    # Check how many keywords from model answer are in student answer
    student_lower = student_answer.lower()
    keyword_matches = 0
    
    for keyword in model_keywords:
        # Check for exact match or close match using Levenshtein distance
        if keyword.lower() in student_lower or any(
            jellyfish.levenshtein_distance(keyword.lower(), word.lower()) <= 2 
            for word in word_tokenize(student_answer) if len(word) > 3
        ):
            keyword_matches += 1
    
    # Keyword coverage ratio
    if not model_keywords:
        keyword_ratio = 0
    else:
        keyword_ratio = keyword_matches / len(model_keywords)
    
    # Length ratio (penalize very short answers)
    student_length = len(word_tokenize(student_answer))
    model_length = len(word_tokenize(model_answer))
    length_ratio = min(student_length / max(model_length * 0.5, 1), 1.0)
    
    # Get reading ease score (higher complexity might indicate deeper knowledge)
    complexity = min(textstat.flesch_reading_ease(student_answer) / 100, 1.0)
    # Invert so higher complexity gives higher score (up to a point)
    complexity_score = 1 - (abs(complexity - 0.4) / 0.6)
    
    # Combine metrics with weights
    depth_score = (
        0.4 * keyword_ratio +
        0.3 * length_ratio +
        0.3 * complexity_score
    )
    
    return min(max(depth_score, 0.0), 1.0)

def coherence_score(text):
    """
    Evaluate the coherence and structure of the answer.
    
    Args:
        text (str): Input text
        
    Returns:
        float: Coherence score (0-1)
    """
    # Parse sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 1:
        return 0.5  # Not enough sentences to evaluate coherence properly
    
    # Analyze sentence structure
    doc = nlp(text)
    
    # Check for proper sentence structure (subject-verb-object)
    sentence_structure_score = 0
    for sent in doc.sents:
        has_subject = False
        has_verb = False
        
        for token in sent:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                has_subject = True
            if token.pos_ == "VERB":
                has_verb = True
                
        if has_subject and has_verb:
            sentence_structure_score += 1
    
    if not sentences:
        sentence_structure_ratio = 0
    else:
        sentence_structure_ratio = sentence_structure_score / len(sentences)
    
    # Calculate transition words ratio
    transition_words = [
        "furthermore", "moreover", "in addition", "additionally", "besides",
        "however", "nevertheless", "on the other hand", "conversely", "although",
        "even though", "while", "despite", "in spite of", "because", "since",
        "as", "therefore", "thus", "consequently", "as a result", "for this reason",
        "first", "second", "third", "finally", "lastly", "in conclusion", "to summarize",
        "for example", "for instance", "specifically", "in particular", "namely",
        "such as", "similarly", "likewise", "in the same way", "by contrast"
    ]
    
    words = word_tokenize(text.lower())
    transition_count = sum(1 for i in range(len(words)-1) 
                           if ' '.join(words[i:i+2]).lower() in transition_words 
                           or words[i].lower() in transition_words)
    
    # Calculate transition ratio based on sentence count
    if len(sentences) <= 1:
        transition_ratio = 0
    else:
        expected_transitions = len(sentences) - 1
        transition_ratio = min(transition_count / expected_transitions, 1.0) if expected_transitions > 0 else 0
    
    # Get paragraph structure score
    paragraphs = text.split('\n\n')
    paragraph_score = min(len(paragraphs) / 3, 1.0)  # Reward having multiple paragraphs up to 3
    
    # Combine scores
    coherence = (
        0.4 * sentence_structure_ratio +
        0.4 * transition_ratio +
        0.2 * paragraph_score
    )
    
    return min(max(coherence, 0.0), 1.0)

def language_quality_score(text):
    """
    Evaluate the language quality of the answer.
    
    Args:
        text (str): Input text
        
    Returns:
        float: Language quality score (0-1)
    """
    # Check for very short text
    if len(text.split()) < 10:
        return 0.3
    
    # Count spelling errors (simplified approach)
    doc = nlp(text)
    spelling_error_count = 0
    total_words = 0
    
    for token in doc:
        if token.is_alpha and len(token.text) > 2:
            total_words += 1
            # Simplified spelling check
            if token.text.lower() not in nlp.vocab:
                spelling_error_count += 1
    
    spelling_accuracy = 1.0 - (spelling_error_count / max(total_words, 1))
    
    # Grammar factors
    sentences = sent_tokenize(text)
    
    # Very simple grammar checks
    grammar_issues = 0
    
    for sentence in sentences:
        # Check for sentence capitalization
        if not sentence.strip()[0].isupper():
            grammar_issues += 1
        
        # Check for proper ending punctuation
        if not sentence.strip()[-1] in ['.', '!', '?']:
            grammar_issues += 1
    
    grammar_score = 1.0 - (grammar_issues / max(len(sentences) * 2, 1))
    
    # Vocabulary diversity
    words = [token.lemma_.lower() for token in doc if token.is_alpha]
    unique_words = set(words)
    
    if not words:
        vocabulary_diversity = 0
    else:
        vocabulary_diversity = min(len(unique_words) / len(words) * 2, 1.0)
    
    # Sentence length variation
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    
    if not sentence_lengths or len(sentence_lengths) == 1:
        sentence_variation = 0.5  # Neutral score for single sentence
    else:
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        # Normalize variation (higher variation is better, up to a point)
        sentence_variation = min(std_length / mean_length * 2, 1.0) if mean_length > 0 else 0
    
    # Combine metrics
    language_score = (
        0.35 * spelling_accuracy +
        0.35 * grammar_score +
        0.20 * vocabulary_diversity +
        0.10 * sentence_variation
    )
    
    return min(max(language_score, 0.0), 1.0)

def keyword_coverage_score(student_answer, model_answer):
    """
    Calculate how well the student answer covers the keywords from the model answer.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model answer
        
    Returns:
        float: Keyword coverage score (0-1)
    """
    # Extract keywords from model answer
    model_keywords = extract_keywords(model_answer, n=20)
    
    if not model_keywords:
        return 0.5  # Default if no keywords extracted
    
    # Check for keyword presence in student answer
    student_lower = student_answer.lower()
    student_doc = nlp(student_answer.lower())
    student_lemmas = {token.lemma_ for token in student_doc if token.is_alpha}
    
    matches = 0
    for keyword in model_keywords:
        keyword_lower = keyword.lower()
        # Check exact match
        if keyword_lower in student_lower:
            matches += 1
            continue
            
        # Check lemmatized match
        keyword_doc = nlp(keyword_lower)
        keyword_lemmas = {token.lemma_ for token in keyword_doc if token.is_alpha}
        
        if any(lemma in student_lemmas for lemma in keyword_lemmas):
            matches += 1
            continue
            
        # Check for fuzzy match using Levenshtein distance
        words = word_tokenize(student_lower)
        if any(jellyfish.levenshtein_distance(keyword_lower, word) <= 2 for word in words if len(word) > 3):
            matches += 0.5  # Partial match
    
    # Calculate coverage ratio
    coverage_ratio = matches / len(model_keywords)
    
    return min(max(coverage_ratio, 0.0), 1.0)

def originality_score(student_answer, model_answer):
    """
    Evaluate the originality of the student answer compared to the model answer.
    
    Args:
        student_answer (str): Student's answer
        model_answer (str): Model answer
        
    Returns:
        float: Originality score (0-1)
    """
    # Get n-grams from both answers
    student_words = word_tokenize(student_answer.lower())
    model_words = word_tokenize(model_answer.lower())
    
    # Use bigrams and trigrams for comparison
    student_bigrams = list(ngrams(student_words, 2))
    model_bigrams = list(ngrams(model_words, 2))
    
    student_trigrams = list(ngrams(student_words, 3))
    model_trigrams = list(ngrams(model_words, 3))
    
    # Calculate overlap ratios
    if not student_bigrams:
        bigram_overlap = 1.0  # Default if no student bigrams
    else:
        common_bigrams = sum(1 for bg in student_bigrams if bg in model_bigrams)
        bigram_overlap = 1.0 - (common_bigrams / len(student_bigrams))
    
    if not student_trigrams:
        trigram_overlap = 1.0  # Default if no student trigrams
    else:
        common_trigrams = sum(1 for tg in student_trigrams if tg in model_trigrams)
        trigram_overlap = 1.0 - (common_trigrams / len(student_trigrams))
    
    # Overall originality score
    originality = (0.4 * bigram_overlap) + (0.6 * trigram_overlap)
    
    # Adjust for very short answers
    if len(student_words) < 20:
        return originality * (len(student_words) / 20)
        
    return min(max(originality, 0.0), 1.0)

if __name__ == "__main__":
    # Test metrics
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
    
    # Get embeddings
    from models.model import load_model, get_embeddings
    model = load_model()
    student_embedding = get_embeddings(model, student_answer)
    model_embedding = get_embeddings(model, model_answer)
    
    # Calculate metrics
    print(f"Content relevance: {content_relevance_score(student_answer, model_answer, student_embedding, model_embedding):.2f}")
    print(f"Knowledge depth: {knowledge_depth_score(student_answer, model_answer, student_embedding, model_embedding):.2f}")
    print(f"Coherence: {coherence_score(student_answer):.2f}")
    print(f"Language quality: {language_quality_score(student_answer):.2f}")
    print(f"Keyword coverage: {keyword_coverage_score(student_answer, model_answer):.2f}")
    print(f"Originality: {originality_score(student_answer, model_answer):.2f}")
