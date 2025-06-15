import os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocess import preprocess_new_text, clean_text

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

# Load model and vectorizer
def load_model_and_vectorizer():
    global model, vectorizer
    
    # Check if model and vectorizer exist
    if not (os.path.exists('models/fake_news_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl')):
        # If not, train the model
        from model_training import train_model
        model, vectorizer = train_model()
    else:
        # Load existing model and vectorizer
        model = joblib.load('models/fake_news_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load models before first request
@app.before_request
def before_request():
    global model, vectorizer
    if model is None or vectorizer is None:
        load_model_and_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from request
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'No text provided',
            'prediction': None,
            'probability': None
        })
    
    # Ensure model and vectorizer are loaded
    global model, vectorizer
    if model is None or vectorizer is None:
        load_model_and_vectorizer()
    
    # Preprocess and vectorize the text
    vectorized_text = preprocess_new_text(text, vectorizer)
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]
    
    # Generate explanation based on prediction and confidence
    def generate_explanation(pred, conf):
        if pred == 'Fake News':
            if conf > 0.9:
                return "This article shows strong indicators of being fake news. The language patterns, structure, and content are highly consistent with known fake news articles."
            elif conf > 0.7:
                return "This article has several characteristics commonly found in fake news. While not absolutely certain, the model has detected concerning patterns."
            else:
                return "This article has some elements that suggest it might be fake news, but the confidence is lower. Consider verifying with trusted sources."
        else:  # Real news
            if conf > 0.9:
                return "This article strongly resembles legitimate news content. The language, structure, and content are consistent with reliable reporting."
            elif conf > 0.7:
                return "This article appears to be legitimate news. While not absolutely certain, the model has detected patterns consistent with real news."
            else:
                return "This article has some characteristics of real news, but the confidence is lower. Consider verifying with trusted sources."
    
    # Extract key terms that influenced the prediction
    def extract_key_terms(text, pred):
        # Clean the text first
        cleaned_text = clean_text(text)
        
        # Get words from the text
        words = cleaned_text.split()
        
        # Common fake news terms (simplified approach)
        fake_indicators = ['hoax', 'conspiracy', 'shocking', 'unbelievable', 'secret', 
                          'they', 'them', 'exposed', 'truth', 'believe', 'incredible']
        
        # Common real news terms (simplified approach)
        real_indicators = ['according', 'reported', 'officials', 'sources', 'study', 
                          'research', 'evidence', 'data', 'experts', 'analysis']
        
        # Find matching terms based on prediction
        if pred == 'Fake News':
            matches = [word for word in words if word in fake_indicators]
        else:
            matches = [word for word in words if word in real_indicators]
        
        # Return unique matches or default message
        unique_matches = list(set(matches))
        return unique_matches[:5] if unique_matches else ["No specific key terms identified"]
    
    # Get prediction confidence
    conf_value = float(probability[1]) if prediction == 1 else float(probability[0])
    pred_result = 'Real News' if prediction == 1 else 'Fake News'
    
    # Format result
    result = {
        'prediction': pred_result,
        'probability': conf_value,
        'confidence': f"{conf_value * 100:.2f}%",
        'percentage': float(conf_value * 100),  # Add percentage for confidence bar width
        'explanation': generate_explanation(pred_result, conf_value),
        'key_terms': extract_key_terms(text, pred_result)
    }
    
    # Return JSON if request was JSON, otherwise render result template
    if request.is_json:
        return jsonify(result)
    else:
        return render_template('result.html', text=text, **result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # API endpoint for programmatic access
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided in request',
            'prediction': None,
            'probability': None
        }), 400
    
    text = data['text']
    
    # Ensure model and vectorizer are loaded
    global model, vectorizer
    if model is None or vectorizer is None:
        load_model_and_vectorizer()
    
    # Preprocess and vectorize the text
    vectorized_text = preprocess_new_text(text, vectorizer)
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]
    
    # Format result
    result = {
        'prediction': 'Real News' if prediction == 1 else 'Fake News',
        'probability': float(probability[1]) if prediction == 1 else float(probability[0]),
        'confidence': f"{max(probability) * 100:.2f}%",
        'percentage': float(max(probability) * 100)  # Add percentage for confidence bar width
    }
    
    return jsonify(result)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Load model and vectorizer at startup
    load_model_and_vectorizer()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)