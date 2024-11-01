# Heart Disease Q&A Chatbot 🫀

A healthcare chatbot specialized in heart disease information using BERT transformer model to provide accurate and helpful responses to user queries about cardiovascular health.

## Overview

This chatbot is designed to provide reliable information about heart diseases, symptoms, treatments, and prevention strategies. It uses a fine-tuned BERT model to understand and respond to user questions with medically accurate information.

## Project Presentation with Demo Video

[Heart-Disease Assistant Chatbot](https://drive.google.com/file/d/1k4fiTbrTkWXwboZI57JF_wZ_SKMmn3PZ/view?usp=sharing)

## Features

- 🔍 Accurate question-answer matching using BERT
- 💊 Comprehensive heart disease information
- 🏥 Medical terminology explanations
- ❤️ Prevention and lifestyle advice
- 🚨 Emergency symptoms awareness
- 📊 Interactive and user-friendly interface

## Tech Stack

- **Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Framework**: TensorFlow 2.x
- **Frontend**: Streamlit
- **Language**: Python 3.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/DavidkingMazimpaka/CHAT-BOT.git
cd CHAT-BOT

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Requirements

```text
tensorflow>=2.6.0
transformers>=4.15.0
streamlit>=1.10.0
pandas>=1.3.0
numpy>=1.19.5
scikit-learn>=0.24.2
```

## Dataset

The Q&A dataset includes:
- Medical questions about heart diseases
- Detailed, accurate answers
- Question patterns for better matching
- Categorical tags for organization

Dataset categories cover:
- Basic heart disease concepts
- Symptoms and diagnosis
- Treatment options
- Prevention strategies
- Lifestyle factors
- Emergency situations
- Risk factors

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Type your heart-related questions in the chat interface

## Project Structure

```
heart-disease-chatbot/
├── app.py                  # Streamlit application
├── model/
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction functions
│   └── utils.py           # Helper functions
├── data/
│   ├── qa_dataset.csv     # Q&A dataset
│   └── process_data.py    # Data preprocessing
├── config/
│   └── config.yaml        # Configuration files
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Model Training

```bash
# Preprocess the dataset
python data/process_data.py

# Train the model
python model/train.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements

- [ ] Add multi-language support
- [ ] Implement confidence scoring
- [ ] Add more specialized medical topics
- [ ] Improve response time
- [ ] Add speech-to-text capability
- [ ] Implement regular dataset updates

## Important Note

This chatbot is designed for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.

## Contact

Your Name - [@DavidkingMazimpaka](mailto:mmazimpakadavid607@gmail.com)

Project Link: [https://github.com/DavidkingMazimpaka/CHAT-BOT](https://github.com/yourusername/heart-disease-chatbot)

---
⚠️ **Disclaimer**: This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.