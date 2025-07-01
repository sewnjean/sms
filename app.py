import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['text'] = df['text'].str.lower()
    return df

# Train model
@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# Load data and train model
df = load_data()
model, vectorizer = train_model(df)

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Message Classifier")
st.markdown("Enter a message below to check whether it's **spam** or **ham** (not spam).")

# Input box
user_input = st.text_area("✍️ Your Message", height=150)

# Predict button
if st.button("🚀 Classify"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        input_processed = user_input.lower()
        input_vec = vectorizer.transform([input_processed])
        prediction = model.predict(input_vec)[0]

        if prediction == "spam":
            st.error("🚫 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM** (not spam).")
