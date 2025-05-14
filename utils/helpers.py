"""
Helper functions for the Theoretical Answer Evaluation System.
"""

import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize

def load_json_data(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
    
    Returns:
        dict: Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not valid JSON")
        return {}

def save_json_data(data, filepath):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Path to the JSON file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def export_results_to_csv(results, filepath):
    """
    Export evaluation results to a CSV file.
    
    Args:
        results (list): List of evaluation results
        filepath (str): Path to the CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert results to dataframe
        df = pd.DataFrame(results)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error exporting results: {e}")
        return False

def generate_report(evaluation_history, output_filepath):
    """
    Generate a detailed report of evaluation history.
    
    Args:
        evaluation_history (list): List of evaluation results
        output_filepath (str): Path to save the report
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create dataframe
        data = []
        
        for item in evaluation_history:
            row = {
                'Question': item['question'],
                'Score': item['score'],
                'Max Score': item['max_score'],
                'Percentage': (item['score'] / item['max_score']) * 100
            }
            
            # Add detailed metrics
            for metric, score in item['metrics'].items():
                row[metric] = score
                
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Generate report
        with open(output_filepath, 'w') as f:
            f.write("# Theoretical Answer Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"Total answers evaluated: {len(df)}\n")
            f.write(f"Average score: {df['Score'].mean():.2f}\n")
            f.write(f"Average percentage: {df['Percentage'].mean():.2f}%\n\n")
            
            f.write("## Detailed Metrics\n\n")
            
            # Get average scores for each metric
            metrics = [col for col in df.columns if col not in ['Question', 'Score', 'Max Score', 'Percentage']]
            
            for metric in metrics:
                f.write(f"- {metric}: {df[metric].mean():.2f}\n")
                
            f.write("\n## Individual Results\n\n")
            
            for _, row in df.iterrows():
                f.write(f"### Question: {row['Question']}\n\n")
                f.write(f"- Score: {row['Score']}/{row['Max Score']} ({row['Percentage']:.2f}%)\n")
                
                for metric in metrics:
                    f.write(f"- {metric}: {row[metric]:.2f}\n")
                    
                f.write("\n")
        
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

def visualize_metrics(evaluation_results):
    """
    Visualize evaluation metrics.
    
    Args:
        evaluation_results (dict): Evaluation results
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Extract metrics
    detailed_scores = evaluation_results["detailed_scores"]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart for detailed scores
    metrics = list(detailed_scores.keys())
    scores = list(detailed_scores.values())
    
    ax1.bar(metrics, scores, color='skyblue')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Detailed Metric Scores')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Radar chart for visualization
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    scores += scores[:1]  # Complete the circle
    
    ax2.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    ax2 = plt.subplot(122, polar=True)
    ax2.fill(angles, scores, color='skyblue', alpha=0.7)
    ax2.set_yticklabels([])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_title('Metric Radar Chart')
    
    plt.tight_layout()
    
    return fig

def count_words(text):
    """
    Count the number of words in text.
    
    Args:
        text (str): Input text
    
    Returns:
        int: Word count
    """
    return len(word_tokenize(text))

def count_sentences(text):
    """
    Count the number of sentences in text.
    
    Args:
        text (str): Input text
    
    Returns:
        int: Sentence count
    """
    return len(sent_tokenize(text))

def normalize_text_for_comparison(text):
    """
    Normalize text for comparison by removing extra spaces, 
    converting to lowercase, etc.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def truncate_text(text, max_length=100, add_ellipsis=True):
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Input text
        max_length (int): Maximum length
        add_ellipsis (bool): Whether to add ellipsis
    
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length].rsplit(' ', 1)[0]
    
    if add_ellipsis:
        truncated += "..."
        
    return truncated

def format_time(seconds):
    """
    Format time in seconds to minutes and seconds.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

if __name__ == "__main__":
    # Example usage
    sample_results = {
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
    
    # Visualize metrics
    fig = visualize_metrics(sample_results)
    plt.show()
    
    # Test word and sentence counting
    sample_text = "This is a sample text. It has two sentences."
    print(f"Word count: {count_words(sample_text)}")
    print(f"Sentence count: {count_sentences(sample_text)}")
