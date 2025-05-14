"""
Text preprocessing utilities for the Theoretical Answer Evaluation System.
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Get stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Basic text cleaning function.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """
    Lemmatize text using WordNet lemmatizer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lemmatized text
    """
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_short_words(text, min_length=2):
    """
    Remove short words from text.
    
    Args:
        text (str): Input text
        min_length (int): Minimum word length to keep
        
    Returns:
        str: Text with short words removed
    """
    words = word_tokenize(text)
    filtered_words = [word for word in words if len(word) >= min_length]
    return ' '.join(filtered_words)

def preprocess_text(text, remove_stops=True, lemmatize=True):
    """
    Full preprocessing pipeline.
    
    Args:
        text (str): Input text
        remove_stops (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize text
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = clean_text(text)
    
    # Remove short words
    text = remove_short_words(text)
    
    # Optionally remove stopwords
    if remove_stops:
        text = remove_stopwords(text)
    
    # Optionally lemmatize
    if lemmatize:
        text = lemmatize_text(text)
    
    return text

def extract_keywords(text, n=10):
    """
    Extract key terms/phrases from text.
    
    Args:
        text (str): Input text
        n (int): Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    doc = nlp(text)
    
    # Extract noun chunks and named entities
    keywords = []
    
    # Add noun chunks
    for chunk in doc.noun_chunks:
        keywords.append(chunk.text)
    
    # Add named entities
    for ent in doc.ents:
        keywords.append(ent.text)
    
    # Process keywords
    cleaned_keywords = []
    for keyword in keywords:
        keyword = clean_text(keyword)
        if keyword and len(keyword.split()) <= 3:  # Keep phrases with at most 3 words
            cleaned_keywords.append(keyword)
    
    # Count frequencies and return top n
    from collections import Counter
    keyword_counts = Counter(cleaned_keywords)
    top_keywords = [kw for kw, _ in keyword_counts.most_common(n)]
    
    return top_keywords

def get_sentence_count(text):
    """
    Count the number of sentences in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of sentences
    """
    doc = nlp(text)
    return len(list(doc.sents))

def get_word_count(text):
    """
    Count the number of words in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of words
    """
    return len(word_tokenize(text))

if __name__ == "__main__":
    # Test the preprocessing functions
    sample_text = "This is a sample text with some stopwords. It also has punctuation and numbers like 123!"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
    print("No stopwords:", remove_stopwords(clean_text(sample_text)))
    print("Lemmatized:", lemmatize_text(clean_text(sample_text)))
    print("Fully preprocessed:", preprocess_text(sample_text))
