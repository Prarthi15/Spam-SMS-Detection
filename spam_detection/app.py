from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize the Flask application
app = Flask(__name__)


# Text preprocessing function
def preprocess_text(message):
    wnl = WordNetLemmatizer()
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=message)
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    return ' '.join(lemm_words)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    preprocessed_message = preprocess_text(message)
    vectorized_message = vectorizer.transform([preprocessed_message])
    
    # Naive Bayes Prediction
    nb_prediction = nb_model.predict(vectorized_message)[0]
    nb_proba = nb_model.predict_proba(vectorized_message)[0]
    
    # Random Forest Prediction
    rf_prediction = rf_model.predict(vectorized_message)[0]
    rf_proba = rf_model.predict_proba(vectorized_message)[0]

    response = {
        'naive_bayes': {
            'prediction': 'spam' if nb_prediction == 1 else 'ham',
            'probability': nb_proba[1] if nb_prediction == 1 else nb_proba[0]
        },
        'random_forest': {
            'prediction': 'spam' if rf_prediction == 1 else 'ham',
            'probability': rf_proba[1] if rf_prediction == 1 else rf_proba[0]
        }
    }
    return render_template('index.html', prediction=response, message=message)

# Run the Flask app
if __name__ == '__main__':
    nltk.download("stopwords")
    nltk.download("wordnet")
    app.run(debug=True)
