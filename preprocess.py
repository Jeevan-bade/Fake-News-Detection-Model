import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    
    Clean and preprocess text data
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ''

def load_and_preprocess_data(fake_news_path, true_news_path):
    """
    Load and preprocess the fake and true news datasets
    """
    # Load datasets
    fake_df = pd.read_csv(fake_news_path)
    true_df = pd.read_csv(true_news_path)
    
    # Add labels
    fake_df['label'] = 0  # 0 for fake news
    true_df['label'] = 1  # 1 for true news
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    return df

def prepare_features(df, text_column='cleaned_text', test_size=0.2, random_state=42):
    """
    Prepare features and split data into training and testing sets
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], 
        df['label'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform testing data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

def preprocess_new_text(text, vectorizer):
    """
    Preprocess new text data for prediction
    """
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    return vectorized

def enhanced_preprocessing(text):
    # Add these features
    return {
        'tfidf': tfidf_features,
        'sentiment': textblob_sentiment,
        'readability': flesch_score,
        'named_entities': len(ner_results)
    }