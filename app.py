from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
try:
    with open('models/fake_news_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please run model_training.py first.")
    model = None
    vectorizer = None

# Sample real news articles for testing
SAMPLE_REAL_NEWS = [
    "The Federal Reserve announced today that it will maintain current interest rates at their meeting. The decision comes after careful consideration of current economic indicators and inflation trends. Fed Chairman emphasized the importance of monitoring economic data closely in the coming months.",
    
    "Scientists at Stanford University have published new research on renewable energy efficiency in solar panels. The study, published in Nature Energy journal, shows a 15% improvement in energy conversion rates using a new silicon compound. The research was peer-reviewed and funded by the Department of Energy.",
    
    "The World Health Organization released its annual report on global health trends today. The comprehensive study covers health outcomes across 194 countries and highlights improvements in maternal mortality rates and vaccination coverage. The data was collected through established health monitoring systems.",
    
    
    "A new archaeological discovery in Egypt has revealed artifacts dating back 3,000 years. The excavation, led by a team from Cairo University in collaboration with international researchers, uncovered pottery and tools that provide insight into daily life during the New Kingdom period."
]

def preprocess_text(text):
    """
    Preprocess text for prediction using the same steps as training
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def get_confidence_explanation(prediction, confidence):
    """
    Generate explanation based on prediction and confidence level
    """
    confidence_num = float(confidence.strip('%'))
    
    if prediction == "Fake News":
        if confidence_num >= 90:
            return "The model is highly confident this text contains characteristics typical of fake news, such as sensational language, unverified claims, or misleading information patterns."
        elif confidence_num >= 70:
            return "The model detected several indicators suggesting this might be fake news, including suspicious language patterns or questionable claims that warrant fact-checking."
        else:
            return "The model has some concerns about this text but with lower confidence. It's recommended to verify the information through reliable sources."
    else:
        if confidence_num >= 90:
            return "The model is highly confident this text appears to be legitimate news, showing characteristics of factual reporting with credible language patterns."
        elif confidence_num >= 70:
            return "The model believes this text shows signs of legitimate news reporting, though additional verification with primary sources is always recommended."
        else:
            return "The model leans toward this being real news but with moderate confidence. Cross-referencing with established news sources would be beneficial."

def extract_key_terms(text, prediction):
    """
    Extract key terms that might have influenced the prediction
    This is a simplified version - in practice, you'd use model feature importance
    """
    # Common fake news indicators
    fake_indicators = [
        'breaking', 'shocking', 'unbelievable', 'scientists discover', 
        'government conspiracy', 'they don\'t want you to know', 'secret',
        'exposed', 'leaked', 'exclusive', 'bombshell', 'scandal'
    ]
    
    # Common real news indicators  
    real_indicators = [
        'according to', 'research shows', 'study published', 'officials said',
        'data indicates', 'report released', 'investigation found', 'announced',
        'statement', 'confirmed', 'university', 'department'
    ]
    
    text_lower = text.lower()
    found_terms = []
    
    if prediction == "Fake News":
        for term in fake_indicators:
            if term in text_lower:
                found_terms.append(term)
    else:
        for term in real_indicators:
            if term in text_lower:
                found_terms.append(term)
    
    # Also look for emotional/sensational words
    emotional_words = ['amazing', 'incredible', 'shocking', 'outrageous', 'devastating']
    for word in emotional_words:
        if word in text_lower and prediction == "Fake News":
            found_terms.append(word)
    
    return found_terms[:5] if found_terms else ["No specific key terms identified"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not available. Please contact administrator.'
            })
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided for analysis.'
            })
        
        text = data['text'].strip()
        
        # Validate input
        if len(text) < 50:
            return jsonify({
                'error': 'Text too short. Please provide at least 50 characters for accurate analysis.'
            })
        
        if len(text) > 10000:
            return jsonify({
                'error': 'Text too long. Please limit to 10,000 characters.'
            })
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return jsonify({
                'error': 'Unable to process the provided text. Please check for valid content.'
            })
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Convert prediction to readable format
        prediction_label = "Fake News" if prediction == 1 else "Real News"
        
        # Calculate confidence (probability of predicted class)
        confidence = max(prediction_proba) * 100
        confidence_str = f"{confidence:.1f}%"
        
        # Generate explanation and key terms
        explanation = get_confidence_explanation(prediction_label, confidence_str)
        key_terms = extract_key_terms(text, prediction_label)
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence_str,
            'percentage': round(confidence, 1),
            'explanation': explanation,
            'key_terms': key_terms
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'An error occurred during analysis. Please try again.'
        })

@app.route('/sample_real')
def sample_real():
    """Return a random sample of real news for testing"""
    try:
        sample = random.choice(SAMPLE_REAL_NEWS)
        return jsonify({'sample': sample})
    except Exception as e:
        return jsonify({'error': 'Failed to load sample news'}), 500

@app.route('/result')
def result():
    """Render result page (if using form submission instead of AJAX)"""
    return render_template('result.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)