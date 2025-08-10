import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load the trained sentiment model
@st.cache_resource
def load_model(path="models/sentiment_pipeline.joblib"):
    return joblib.load(path)

model = load_model()

# App title
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")
st.title("💬 Customer Review Sentiment Analysis")
st.markdown("Enter a customer review below and let the model predict the sentiment.")

# User input
review_text = st.text_area("✏️ Your Review", height=150, placeholder="Type your review here...")

# Predict button
if st.button("🔍 Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review text!")
    else:
        pred = model.predict([review_text])[0]
        probs = model.predict_proba([review_text])[0]

        # Sentiment colors
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
        st.markdown(f"**Prediction:** <span style='color:{color_map.get(pred, 'black')}; font-size: 20px;'>{pred}</span>", unsafe_allow_html=True)

        # Probability DataFrame
        df_probs = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        # Display table
        st.subheader("📊 Sentiment Probabilities")
        fig = px.bar(df_probs, x="Sentiment", y="Probability", color="Sentiment", 
                     color_discrete_map=color_map, range_y=[0,1], text="Probability")
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
