from flask import Flask, render_template, request, jsonify
import os
import pickle

app = Flask(__name__)

with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    
    
    email_vectorized = vectorizer.transform([email_text])
    
    
    prediction = model.predict(email_vectorized)[0]
    
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
