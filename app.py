
import streamlit as st
import torch
import json
import os
import sqlite3
import uuid

from huggingface_hub import login
login(token="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")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline

# ----------------------------------------
# LOAD MODELS
# ----------------------------------------
bert_model_path = "./saved_model"
label_mapping_path = os.path.join(bert_model_path, "label_mapping.json")

# Load BERT model and tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
bert_model.eval()

# Load label mapping
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)
id2label = {v: k for k, v in label_mapping.items()}

# Load Mistral for explanation
mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer)

# ----------------------------------------
# DATABASE SETUP
# ----------------------------------------
conn = sqlite3.connect("feedback.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    name TEXT,
    age INTEGER,
    user_class TEXT,
    symptoms TEXT,
    disease TEXT,
    explanation TEXT,
    understandability INTEGER,
    satisfaction INTEGER
)
''')
conn.commit()

# ----------------------------------------
# FUNCTIONS
# ----------------------------------------
def predict_disease(symptom_text):
    inputs = bert_tokenizer(symptom_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predicted_class_id = outputs.logits.argmax(dim=-1).item()
    return id2label.get(predicted_class_id, f"Unknown (class {predicted_class_id})")

def generate_explanation(disease, audience):
    prompt = f"[INST] Briefly explain the disease {disease} to a {audience} in no more than 3 lines. Use clear and simple language. [/INST]"
    result = mistral_pipe(prompt, max_new_tokens=70, do_sample=True, temperature=0.7, repetition_penalty=1.1)
    return result[0]["generated_text"].replace(prompt, "").strip()

def insert_feedback(entry):
    c.execute('''
    INSERT INTO feedback (id, name, age, user_class, symptoms, disease, explanation, understandability, satisfaction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry["id"], entry["name"], entry["age"], entry["user_class"], entry["symptoms"],
        entry["disease"], entry["explanation"], entry["understandability"], entry["satisfaction"]
    ))
    conn.commit()

def get_user_feedback(uid):
    c.execute("SELECT * FROM feedback WHERE id = ?", (uid,))
    return c.fetchall()

def get_all_feedback():
    c.execute("SELECT * FROM feedback")
    return c.fetchall()

# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------
st.set_page_config(page_title="🧠 Disease Predictor", layout="centered")

st.title("🧠 AI-Powered Disease Prediction & Explanation")
st.write("Enter your symptoms and receive an AI-generated prediction and explanation.")

# User input
with st.form("user_form"):
    name = st.text_input("Your Name")
    age = st.number_input("Your Age", min_value=1, max_value=120)
    user_class = st.selectbox("You are a...", ["student", "elderly", "doctor"])
    symptoms = st.text_area("Describe your symptoms")
    submitted = st.form_submit_button("Predict")

if submitted and name and symptoms:
    with st.spinner("Analyzing..."):
        disease = predict_disease(symptoms)
        explanation = generate_explanation(disease, user_class)
        uid = str(uuid.uuid4())

    st.success(f"**Prediction:** {disease}")
    st.info(f"**Explanation for {user_class}:** {explanation}")

    # Feedback collection
    with st.form("feedback_form"):
        st.markdown("### 🤔 Your Feedback")
        understandability = st.slider("Was the explanation easy to understand?", 0, 10, 5)
        satisfaction = st.slider("How satisfied are you with the AI's prediction?", 0, 10, 5)
        feedback_submitted = st.form_submit_button("Submit Feedback")

        if feedback_submitted:
            insert_feedback({
                "id": uid,
                "name": name,
                "age": age,
                "user_class": user_class,
                "symptoms": symptoms,
                "disease": disease,
                "explanation": explanation,
                "understandability": understandability,
                "satisfaction": satisfaction
            })
            st.success(f"✅ Feedback submitted! Your Registration ID is `{uid}`")

# Admin Section
with st.expander("🔒 Admin Panel"):
    admin_code = st.text_input("Enter admin code to view all feedback", type="password")
    if admin_code == "admin123":  # Change this password for security
        st.success("Access granted.")
        all_data = get_all_feedback()
        if all_data:
            st.write("🗂️ All Feedback Entries")
            st.dataframe(all_data)
        else:
            st.info("No feedback submitted yet.")

