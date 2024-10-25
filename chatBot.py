import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import chatBot as st
from streamlit_chat import message

# Load the pre-trained model and supporting files
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)
model = load_model('heartDiseaseChatBot.h5')
with open('HeartDiseases_QA.json', 'r') as f:
    intents = json.load(f)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess the user input
def preprocess_input(text):
    sentence_words = text.lower().split()
    sentence_words = [lemmatizer.lemmatize(word.strip()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array([bag])

# Function to get the response from the model
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tags'] == tag:
            responses = intent['responses']
            return random.choice(responses)
    return "I'm sorry, I don't have any information about that."

# Streamlit app
st.title("Heart-Diseases Chatbot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

user_input = get_text()
if st.button("Send"):
    st.session_state.past.append(user_input)
    preprocessed_input = preprocess_input(user_input)
    result = model.predict(preprocessed_input)[0]
    response_index = np.argmax(result)
    tag = classes[response_index]
    response = get_response([{'intent': tag, 'probability': str(result[response_index])}], intents)
    st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')