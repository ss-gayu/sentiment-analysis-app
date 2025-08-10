# Customer Review Sentiment Analysis

## Overview
This project classifies customer reviews into **positive**, **negative**, or **neutral** using Natural Language Processing (NLP) and Logistic Regression.  
It includes:
- Data preprocessing (text cleaning, tokenization, lemmatization)
- Feature extraction with TF-IDF
- Model training and evaluation
- Streamlit web app for real-time predictions

## Project Structure
sentiment-project/
├── data/                # Raw and processed data
├── notebooks/           # EDA and prototyping
├── src/                 # Source code
├── models/              # Saved models
├── requirements.txt
├── README.md
└── Dockerfile

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/train.py
```

3. Launch the web app:
```bash
streamlit run src/app.py
```
