import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load tokenizer and LSTM model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('LSTM_model.h5')

# Class labels mapping
class_labels = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sad'}

# Preprocess function to convert input text to sequences
def preprocess_text(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    return padded_seq

# Streamlit app
def main():
    st.title("Text Classification with LSTM")

    # Input text from user
    user_input = st.text_area("Enter text for classification:", "")

    if st.button("Classify"):
        if user_input:
            # Preprocess input text
            processed_input = preprocess_text(user_input, tokenizer)
            
            # Get prediction
            prediction = model.predict(processed_input)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Map numerical class to actual class label
            predicted_label = class_labels.get(predicted_class, "Unknown")

            # Display result
            st.write(f"Predicted Class: {predicted_label}")
        else:
            st.write("Please enter some text to classify.")

if __name__ == "__main__":
    main()
