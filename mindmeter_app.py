
import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_model = load_emotion_model()

st.title("🧠 MindMeter: Daily Mood & Productivity Tracker")

user_input = st.text_area("📝 Write your daily journal entry here:")

if st.button("Analyze Entry"):
    if user_input.strip() == "":
        st.warning("Please enter something first!")
    else:
        prediction = emotion_model(user_input)[0]
        emotion = prediction['label']
        score = prediction['score']

        def classify_productivity(emotion):
            if emotion in ['joy', 'surprise']:
                return 'Productive'
            elif emotion in ['sadness', 'fear']:
                return 'Low'
            elif emotion == 'anger':
                return 'Stressed'
            else:
                return 'Neutral'

        productivity = classify_productivity(emotion)

        st.markdown(f"### 🔍 Predicted Emotion: **{emotion.capitalize()}**")
        st.markdown(f"### 🚀 Productivity Level: **{productivity}**")
        st.markdown(f"🧪 Confidence Score: `{score:.2f}`")

        log_data = {
            "Date": [datetime.now().strftime("%Y-%m-%d")],
            "Entry": [user_input],
            "Emotion": [emotion],
            "Productivity_Level": [productivity],
            "Confidence": [score]
        }

        df = pd.DataFrame(log_data)

        try:
            existing = pd.read_csv("mindmeter_log.csv")
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass

        df.to_csv("mindmeter_log.csv", index=False)
        st.success("Your entry has been saved!")

        with st.expander("📜 View Full Log"):
            st.dataframe(df.tail(10))
