import numpy as np
import streamlit as st
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Assistant",
    page_icon="🫀"
)

# Add custom CSS for avatar styling
st.markdown("""
<style>
.chat-container {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 10px;
    padding: 10px;
}
.avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}
.user-avatar {
    background-color: #E0E0E0;
}
.assistant-avatar {
    background-color: #FF4B4B;
}
.message {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Load the JSON data and convert to DataFrame
@st.cache_resource
def load_data():
    with open('HeartDiseases_QA.json', 'r') as f:
        data = json.load(f)
    
    # Convert JSON to DataFrame format
    df = pd.DataFrame(data)
    return df

# Preprocess the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Display function for messages
def display_message(is_user, message, key):
    col1, col2 = st.columns([1, 9])
    
    with col1:
        if is_user:
            st.markdown(
                f"""
                <div class="avatar user-avatar">
                    👤
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="avatar assistant-avatar">
                    🫀
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with col2:
        st.text_area(
            label="" if is_user else "",
            value=message,
            height=None if is_user else 100,
            key=key,
            disabled=True
        )

# Initialize BERT model and tokenizer
tokenizer, model = load_model_and_tokenizer()
df = load_data()

# Preprocess the questions
df['Questions'] = df['Questions'].apply(clean_text)
df['Answers'] = df['Answers'].apply(clean_text)

# Generate embeddings for all questions
@st.cache_resource
def generate_question_embeddings():
    encoded_questions = tokenizer(df['Questions'].tolist(), 
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt")
    
    with torch.no_grad():
        question_embeddings = model(**encoded_questions).last_hidden_state.mean(dim=1)
    return question_embeddings

question_embeddings = generate_question_embeddings()

def get_most_similar_answer(question, question_embeddings, df):
    """Get the most similar answer using BERT embeddings and cosine similarity"""
    try:
        # Clean and encode the input question
        cleaned_question = clean_text(question)
        encoded_question = tokenizer(cleaned_question, 
                                   return_tensors="pt", 
                                   padding=True, 
                                   truncation=True)
        
        with torch.no_grad():
            question_embedding = model(**encoded_question).last_hidden_state.mean(dim=1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_embedding, question_embeddings).flatten()
        
        # Find the index of the most similar question
        most_similar_idx = similarities.argmax()
        
        # Return the corresponding answer
        return df['Answers'].iloc[most_similar_idx]
    
    except Exception as e:
        st.error(f"Error in getting response: {str(e)}")
        return "I'm having trouble processing your request. Please try again."

# Streamlit app
st.title("Heart Disease Assistant ChatBot🫀")

# Initialize session state for chat history
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Input handling
user_input = st.text_input("Ask a question about heart health:", key="input")

if st.button("Send") and user_input:
    st.session_state.past.append(user_input)
    
    # Get response
    response = get_most_similar_answer(user_input, question_embeddings, df)
    st.session_state.generated.append(response)

# Display conversation
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        # Display assistant message
        display_message(
            is_user=False,
            message=st.session_state["generated"][i],
            key=f"assistant_{i}"
        )
        
        # Display user message
        display_message(
            is_user=True,
            message=st.session_state['past'][i],
            key=f"user_{i}"
        )

# Add a separator
st.markdown("---")

# Add footer
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 Ask me anything about heart health!</p>
</div>
""", unsafe_allow_html=True)