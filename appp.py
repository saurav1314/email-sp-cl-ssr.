import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_message(message):
    # Lowercase
    message = message.lower()
    # Tokenize
    words = nltk.word_tokenize(message)

    # Remove alphanumeric and stopwords
    filtered_words = []
    for word in words:
        if word.isalnum():
            if word not in stopwords.words('english') and word not in string.punctuation:
                filtered_words.append(ps.stem(word))  # Stemming

    return " ".join(filtered_words)

# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App UI
st.title("📧 Email Spam Classifier")

# Input box
input_mail = st.text_area("Enter your email content here:")

# Predict button
if st.button('Predict'):
    # Step 1: Preprocess input
    transformed_mail = transform_message(input_mail)

    # Step 2: Vectorize
    vector_input = tfidf.transform([transformed_mail])

    # Step 3: Predict
    result = model.predict(vector_input.toarray())[0]

    # Step 4: Output
    if result == 1:
        st.error("🚫 This is a SPAM message.")
    else:
        st.success("✅ This is NOT a spam message.")