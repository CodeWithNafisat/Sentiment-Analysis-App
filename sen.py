import streamlit as st
import joblib
import re
import pandas as pd
import contractions as con
import spacy
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https\S+|www\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = con.fix(text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and len(t.text) > 2]
    return " ".join(tokens)

# Cached model loading
@st.cache_resource
def load_models():
    log_model = joblib.load("regression_sentiment.pkl")
    svm_model = joblib.load("svm_sentiment.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    reduced_dim = joblib.load("svd.pkl")

    return log_model, svm_model, vectorizer, reduced_dim

log_model, svm_model, vectorizer, reduced_dim = load_models()

# App config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

#  SESSION STATE (Track progress)
if "started" not in st.session_state:
    st.session_state.started = False

# Home page
if not st.session_state.started:
    st.title("Welcome to the Sentiment Analysis App!")
    st.markdown("""
    ### Hello and Welcome!
    You can:
    - Learn more about this web app on the **About Page**
    - Analyze real text or CSV files on the **Main Page**

    ---
    Click below to start exploring 👇
    """)
    if st.button("Get Started"):
        st.session_state.started = True
        st.rerun()

# Main interface (After user clicks Start) 
else:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["About", "Main Page"])

    # About page
    if page == "About":
        st.title("About this App")
        st.write("""
        This Sentiment Analysis App determines whether a movie review 
        is **Positive** or **Negative** using **NLP** and **Machine Learning**.

        ### Features
        - Cleans and preprocesses text automatically  
        - Choose between Logistic Regression and SVM models  
        - Analyze single texts or upload CSVs  
        - Displays confidence scores for each result
        """)

    # MAIN PAGE
    elif page == "Main Page":
        st.title("Sentiment Analysis Dashboard")
        st.write("Enter text or upload a CSV file to analyze sentiment.")

        model_choice = st.selectbox("Select Model:", ["Logistic Regression", "Support Vector Machine"])
        input_mode = st.radio("Select input mode:", ["Single Text", "Upload CSV"])

        # Single text mode
        if input_mode == "Single Text":
            user_text = st.text_area("Enter text:", height=150)
            if st.button("Predict Sentiment"):
                if not user_text.strip():
                    st.warning("Please enter some text.")
                else:
                    clean = clean_text(user_text)
                    vec = vectorizer.transform([clean])
                    svd = reduced_dim.transform(vec)

                    model = log_model if model_choice == "Logistic Regression" else svm_model
                    prediction = model.predict(svd)[0]

                    if hasattr(model, "predict_proba"):
                        confidence = model.predict_proba(svd)[0].max()
                    else:
                        decision = model.decision_function(svd)
                        confidence = 1 / (1 + np.exp(-np.abs(decision[0])))

                    if prediction == 1:
                        st.success(f"Positive Sentiment (Confidence: {confidence:.2f})")
                    else:
                        st.error(f"Negative Sentiment (Confidence: {confidence:.2f})")

        # --- UPLOAD CSV MODE ---
        elif input_mode == "Upload CSV":
            file = st.file_uploader("Upload CSV file", type=["csv"])
            if file is not None:
                df = pd.read_csv(file)
                text_col = st.selectbox("Select the text column:", df.columns)
                if st.button("Run Batch Prediction"):
                    with st.spinner("Analyzing Sentiments..."):
                        df["clean_text"] = df[text_col].astype(str).apply(clean_text)
                        X_vec = vectorizer.transform(df["clean_text"])
                        svd = reduced_dim.transform(X_vec)

                        model = log_model if model_choice == "Logistic Regression" else svm_model
                        preds = model.predict(svd)

                        if hasattr(model, "predict_proba"):
                            confs = model.predict_proba(svd).max(axis=1)
                        else:
                            decision = model.decision_function(svd)
                            confs = 1 / (1 + np.exp(-np.abs(decision)))

                        df["Sentiment"] = np.where(preds == 1, "Positive", "Negative")
                        df["Confidence"] = (confs * 100).round(1).astype(str) + "%"

                        st.success("Done!")
                        st.dataframe(df[["clean_text", "Sentiment", "Confidence"]])

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")
