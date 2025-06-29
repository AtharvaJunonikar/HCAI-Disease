
#  Disease Prediction Dashboard

This app uses a fine-tuned BERT model to predict diseases from symptoms, and generates explanations using Mistral.

## Features
- Inputs: Name, Age, User Class, Symptoms
- Disease prediction (BERT)
- Explanation (Mistral)
- Feedback (Understandability + Satisfaction, 0–10)
- SQLite database with unique Registration ID
- Admin-only panel for viewing/downloading feedback

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py

