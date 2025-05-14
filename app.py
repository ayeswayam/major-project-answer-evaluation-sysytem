import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import io

# Import custom modules
from preprocessing.text_processor import preprocess_text
from evaluation.evaluator import evaluate_answer
from evaluation.feedback import generate_feedback
from utils.ocr import extract_text_from_image
from models.model import load_model, get_embeddings

# Set page config
st.set_page_config(
    page_title="Theoretical Answer Evaluation System",
    page_icon="📝",
    layout="wide"
)

# Initialize session state variables
if 'eval_history' not in st.session_state:
    st.session_state.eval_history = []

if 'model' not in st.session_state:
    try:
        st.session_state.model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.model = None

# Load sample questions
def load_sample_questions():
    try:
        with open('data/sample_questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return some default questions if file not found
        return {
            "questions": [
                {
                    "id": 1,
                    "question": "Explain the concept of inheritance in object-oriented programming.",
                    "model_answer": "Inheritance is a fundamental concept in object-oriented programming that allows a class to inherit properties and methods from another class. The class that inherits is called the subclass or derived class, and the class being inherited from is called the superclass or base class. This promotes code reusability and establishes a relationship between classes, enabling the creation of hierarchical class structures. Through inheritance, subclasses can extend or override functionality of the superclass, implementing the concept of polymorphism.",
                    "max_score": 10
                },
                {
                    "id": 2,
                    "question": "Describe the importance of preprocessing in machine learning pipelines.",
                    "model_answer": "Preprocessing is a critical step in machine learning pipelines that involves transforming raw data into a suitable format for modeling. It includes tasks such as data cleaning, handling missing values, normalization, feature scaling, encoding categorical variables, and feature selection. Proper preprocessing leads to better model performance by reducing noise, removing irrelevant information, and ensuring that the data meets the assumptions of the chosen algorithm. It also helps in addressing issues like class imbalance, outliers, and dimensionality reduction, ultimately improving the generalization capability of the model and preventing potential biases in the results.",
                    "max_score": 10
                }
            ]
        }

# Function to display the evaluation history
def show_evaluation_history():
    if not st.session_state.eval_history:
        st.info("No evaluation history available.")
        return
    
    st.subheader("Evaluation History")
    
    for idx, item in enumerate(st.session_state.eval_history):
        with st.expander(f"Evaluation #{idx+1} - Score: {item['score']}/{item['max_score']} ({(item['score']/item['max_score']*100):.1f}%)"):
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Student Answer:** {item['student_answer']}")
            st.write(f"**Model Answer:** {item['model_answer']}")
            st.write(f"**Score:** {item['score']}/{item['max_score']}")
            
            # Show detailed scores
            st.write("**Detailed Assessment:**")
            metrics_df = pd.DataFrame(item['metrics'].items(), columns=['Metric', 'Score'])
            st.dataframe(metrics_df)
            
            # Show feedback
            st.write(f"**Feedback:** {item['feedback']}")

# Main application
def main():
    st.title("🎓 Theoretical Answer Evaluation System")
    
    st.markdown("""
    This system evaluates subjective answers using NLP techniques and provides detailed feedback.
    Upload an image of a handwritten answer or type the answer directly to get started.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Evaluate Answer", "View History", "About"])
    
    # Load sample questions
    questions = load_sample_questions()['questions']
    
    if page == "Evaluate Answer":
        st.header("Answer Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Question selection
            selected_question = st.selectbox(
                "Select a question:", 
                options=[q["question"] for q in questions],
                index=0
            )
            
            # Get the selected question details
            question_info = next((q for q in questions if q["question"] == selected_question), None)
            
            # Input method selection
            input_method = st.radio("Input Method:", ["Type Answer", "Upload Image"])
            
            student_answer = ""
            if input_method == "Type Answer":
                student_answer = st.text_area("Enter your answer:", height=200)
            else:
                uploaded_file = st.file_uploader("Upload an image of your answer", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Answer", use_column_width=True)
                    
                    # Extract text from image
                    try:
                        with st.spinner("Extracting text from image..."):
                            student_answer = extract_text_from_image(image)
                        st.success("Text extracted successfully!")
                        st.write("**Extracted Text:**")
                        st.write(student_answer)
                    except Exception as e:
                        st.error(f"Error extracting text: {e}")
            
            if st.button("Evaluate Answer"):
                if student_answer and question_info:
                    with st.spinner("Evaluating answer..."):
                        # Preprocess text
                        preprocessed_answer = preprocess_text(student_answer)
                        
                        # Evaluate answer
                        evaluation_result = evaluate_answer(
                            student_answer=preprocessed_answer,
                            model_answer=question_info["model_answer"],
                            max_score=question_info["max_score"],
                            model=st.session_state.model
                        )
                        
                        # Generate feedback
                        feedback = generate_feedback(
                            student_answer=preprocessed_answer,
                            model_answer=question_info["model_answer"],
                            evaluation_result=evaluation_result
                        )
                        
                        # Save to history
                        history_item = {
                            "question": question_info["question"],
                            "student_answer": student_answer,
                            "model_answer": question_info["model_answer"],
                            "score": evaluation_result["overall_score"],
                            "max_score": question_info["max_score"],
                            "metrics": evaluation_result["detailed_scores"],
                            "feedback": feedback
                        }
                        st.session_state.eval_history.append(history_item)
                        
                        # Show results
                        st.success(f"Evaluation completed! Score: {evaluation_result['overall_score']}/{question_info['max_score']} ({evaluation_result['overall_score']/question_info['max_score']*100:.1f}%)")
                else:
                    st.warning("Please enter or upload an answer before evaluation.")
        
        with col2:
            st.subheader("Model Answer")
            if question_info:
                st.write(question_info["model_answer"])
            
            # Display the most recent evaluation result if available
            if st.session_state.eval_history:
                latest = st.session_state.eval_history[-1]
                st.subheader("Evaluation Result")
                st.write(f"**Score:** {latest['score']}/{latest['max_score']} ({latest['score']/latest['max_score']*100:.1f}%)")
                
                # Show detailed scores
                st.write("**Detailed Assessment:**")
                metrics_df = pd.DataFrame(latest['metrics'].items(), columns=['Metric', 'Score'])
                st.dataframe(metrics_df)
                
                # Visualization of scores
                st.write("**Score Breakdown:**")
                fig_data = metrics_df.copy()
                fig_data.set_index('Metric', inplace=True)
                st.bar_chart(fig_data)
                
                # Show feedback
                st.write("**Feedback:**")
                st.write(latest['feedback'])
    
    elif page == "View History":
        show_evaluation_history()
    
    else:  # About page
        st.header("About the System")
        st.markdown("""
        ## Theoretical Answer Evaluation System
        
        This system uses Natural Language Processing (NLP) and Machine Learning techniques to evaluate subjective answers. It analyzes various aspects of the answer including:
        
        - **Content Relevance**: How well the answer addresses the question
        - **Knowledge Depth**: Demonstration of understanding the concepts
        - **Coherence**: Logical flow and structure of the answer
        - **Language Quality**: Grammar, vocabulary, and expression
        
        ### How It Works
        
        1. The system preprocesses the student's answer to normalize text
        2. It extracts semantic features using pre-trained language models
        3. These features are compared with a model answer using multiple similarity metrics
        4. A weighted scoring algorithm calculates the final score
        5. Detailed feedback is generated to help students improve
        
        ### Technology Stack
        
        - **NLP Models**: BERT, Sentence-BERT
        - **Text Processing**: NLTK, SpaCy
        - **OCR**: PyTesseract
        - **Frontend**: Streamlit
        - **Deployment**: Ploomber AI
        
        Developed by Swayam Singh (202100144) under the supervision of Mr. Sital Sharma, Asst. Professor, AI&DS, SMIT, SMU.
        """)

if __name__ == "__main__":
    main()
