import streamlit as st
import tensorflow as tf
import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():

    simple_rnn_model = tf.keras.models.load_model(
        "simple_rnn_model.keras"
    )

    lstm_model = tf.keras.models.load_model(
        "lstm_model.keras"
    )

    gru_model = tf.keras.models.load_model(
        "gru_model.keras"
    )

    return simple_rnn_model, lstm_model, gru_model


@st.cache_resource
def load_tokenizer():

    with open("tokenizer.json", "r") as f:

        tokenizer_json = f.read()

    tokenizer = tokenizer_from_json(
        tokenizer_json
    )

    return tokenizer


simple_rnn_model, lstm_model, gru_model = load_models()
tokenizer = load_tokenizer()

MAX_LENGTH = 200

# =========================
# PREPROCESSING
# =========================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded


# =========================
# PREDICTION FUNCTION
# =========================

def predict_sentiment(model, review):

    processed = preprocess_text(review)

    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]

    positive_prob = float(prediction)

    negative_prob = float(
        1 - prediction
    )

    sentiment = (
        "Positive"
        if prediction >= 0.5
        else "Negative"
    )

    confidence = max(
        positive_prob,
        negative_prob
    )

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )


# =========================
# HEADER
# =========================

st.title(
    "🎬 Movie Review Sentiment Analysis System"
)

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# =========================
# MODEL SELECTION
# =========================

selected_model = st.selectbox(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# =========================
# INPUT AREA
# =========================

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":

            result = predict_sentiment(
                simple_rnn_model,
                review
            )

        elif selected_model == "LSTM":

            result = predict_sentiment(
                lstm_model,
                review
            )

        else:

            result = predict_sentiment(
                gru_model,
                review
            )

        sentiment, confidence, pos_prob, neg_prob = result

        # =========================
        # OUTPUT AREA
        # =========================

        st.markdown("## Prediction")

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        # =========================
        # PROBABILITY TABLE
        # =========================

        st.markdown("## Probabilities")

        probability_df = pd.DataFrame({

            "Class": [
                "Positive",
                "Negative"
            ],

            "Probability": [
                round(pos_prob*100,2),
                round(neg_prob*100,2)
            ]
        })

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        # =========================
        # CONFIDENCE CHART
        # =========================

        st.markdown(
            "## Confidence Chart"
        )

        fig, ax = plt.subplots(
            figsize=(6,4)
        )

        ax.bar(
            ["Positive", "Negative"],
            [pos_prob, neg_prob]
        )

        ax.set_ylim(0,1)

        ax.set_ylabel(
            "Probability"
        )

        ax.set_title(
            "Positive vs Negative Probability"
        )

        st.pyplot(fig)

        # =========================
        # ALL MODEL COMPARISON
        # =========================

        st.markdown("---")

        st.markdown(
            "## Compare All Models"
        )

        rnn_result = predict_sentiment(
            simple_rnn_model,
            review
        )

        lstm_result = predict_sentiment(
            lstm_model,
            review
        )

        gru_result = predict_sentiment(
            gru_model,
            review
        )

        comparison_df = pd.DataFrame({

            "Model": [
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],

            "Sentiment": [
                rnn_result[0],
                lstm_result[0],
                gru_result[0]
            ],

            "Confidence (%)": [

                round(
                    rnn_result[1]*100,
                    2
                ),

                round(
                    lstm_result[1]*100,
                    2
                ),

                round(
                    gru_result[1]*100,
                    2
                )
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # =========================
        # MODEL CONFIDENCE GRAPH
        # =========================

        fig2, ax2 = plt.subplots(
            figsize=(7,4)
        )

        models = [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ]

        confidence_scores = [

            rnn_result[1]*100,

            lstm_result[1]*100,

            gru_result[1]*100
        ]

        ax2.bar(
            models,
            confidence_scores
        )

        ax2.set_ylabel(
            "Confidence (%)"
        )

        ax2.set_title(
            "Model Confidence Comparison"
        )

        st.pyplot(fig2)