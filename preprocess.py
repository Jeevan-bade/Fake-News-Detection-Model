import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Basic text cleaning
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text
    """
    if not text:
        return ""
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_words)

def stem_text(text):
    """
    Apply stemming to text
    """
    if not text:
        return ""
    
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)

def preprocess_text(text):
    """
    Complete text preprocessing pipeline
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text or pd.isna(text):
        return ""
    
    # Step 1: Basic cleaning
    text = clean_text(text)
    
    # Step 2: Remove stopwords
    text = remove_stopwords(text)
    
    # Step 3: Apply stemming
    text = stem_text(text)
    
    # Step 4: Final cleanup
    text = ' '.join(text.split())  # Remove extra spaces
    
    return text

def preprocess_dataframe(df, text_column, target_column=None):
    """
    Preprocess a pandas DataFrame containing news articles
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the column containing text
        target_column (str): Name of the target column (optional)
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df_processed = df.copy()
    
    print(f"Preprocessing {len(df)} articles...")
    
    # Apply preprocessing to text column
    df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(preprocess_text)
    
    # Remove rows with empty processed text
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed[f'{text_column}_processed'].str.len() > 0]
    final_count = len(df_processed)
    
    print(f"Removed {initial_count - final_count} rows with empty processed text")
    print(f"Final dataset size: {final_count}")
    
    if target_column and target_column in df_processed.columns:
        print("Target distribution:")
        print(df_processed[target_column].value_counts())
    
    return df_processed

def extract_features(texts, max_features=10000):
    """
    Extract basic text features for analysis
    
    Args:
        texts (list): List of preprocessed texts
        max_features (int): Maximum number of features to extract
        
    Returns:
        dict: Dictionary containing feature information
    """
    features = {
        'text_lengths': [len(text.split()) for text in texts],
        'char_lengths': [len(text) for text in texts],
        'avg_word_length': [np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in texts]
    }
    
    # Calculate vocabulary
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    vocab = set(all_words)
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Most common words
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]
    
    features['vocabulary_size'] = len(vocab)
    features['total_words'] = len(all_words)
    features['most_common_words'] = most_common[:20]  # Top 20 for display
    
    return features

def validate_preprocessing(original_text, processed_text):
    """
    Validate preprocessing results
    
    Args:
        original_text (str): Original text
        processed_text (str): Processed text
        
    Returns:
        dict: Validation metrics
    """
    validation = {
        'original_length': len(original_text) if original_text else 0,
        'processed_length': len(processed_text) if processed_text else 0,
        'original_words': len(original_text.split()) if original_text else 0,
        'processed_words': len(processed_text.split()) if processed_text else 0,
        'reduction_ratio': 0
    }
    
    if validation['original_words'] > 0:
        validation['reduction_ratio'] = 1 - (validation['processed_words'] / validation['original_words'])
    
    return validation

def preprocess_for_prediction(text):
    """
    Preprocess text specifically for model prediction
    This function should match exactly the preprocessing used during training
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text ready for vectorization
    """
    return preprocess_text(text)

def batch_preprocess(texts, batch_size=1000):
    """
    Process large amounts of text in batches
    
    Args:
        texts (list): List of texts to process
        batch_size (int): Size of each batch
        
    Returns:
        list: List of preprocessed texts
    """
    processed_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_processed = [preprocess_text(text) for text in batch]
        processed_texts.extend(batch_processed)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch)} / {len(texts)} texts")
    
    return processed_texts

# Additional utility functions for specific preprocessing needs

def handle_contractions(text):
    """
    Expand contractions in text
    """
    contractions_dict = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot",
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will",
        "i'm": "i am", "i've": "i have", "isn't": "is not",
        "it'd": "it would", "it'll": "it will", "it's": "it is",
        "let's": "let us", "mightn't": "might not", "mustn't": "must not",
        "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will",
        "they're": "they are", "they've": "they have", "we'd": "we would",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what's": "what is", "where's": "where is", "who's": "who is",
        "won't": "will not", "wouldn't": "would not", "you'd": "you would",
        "you'll": "you will", "you're": "you are", "you've": "you have"
    }
    
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    for contraction, expansion in contractions_dict.items():
        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text_lower)
    
    return text_lower

def remove_duplicate_spaces(text):
    """
    Remove duplicate spaces and normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def filter_by_length(texts, min_length=10, max_length=10000):
    """
    Filter texts by length constraints
    
    Args:
        texts (list): List of texts to filter
        min_length (int): Minimum character length
        max_length (int): Maximum character length
        
    Returns:
        list: Filtered texts
    """
    filtered_texts = []
    
    for text in texts:
        if isinstance(text, str) and min_length <= len(text) <= max_length:
            filtered_texts.append(text)
    
    print(f"Filtered {len(texts)} texts to {len(filtered_texts)} texts")
    print(f"Removed {len(texts) - len(filtered_texts)} texts outside length range [{min_length}, {max_length}]")
    
    return filtered_texts

def detect_language(text):
    """
    Simple language detection (basic implementation)
    Returns True if text appears to be English
    """
    if not isinstance(text, str) or len(text) < 10:
        return False
    
    # Common English words
    english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
        'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
        'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
        'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
        'one', 'all', 'would', 'there', 'their', 'what', 'so',
        'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
    }
    
    words = text.lower().split()
    english_word_count = sum(1 for word in words if word in english_words)
    
    # If at least 30% of words are common English words, consider it English
    if len(words) > 0:
        english_ratio = english_word_count / len(words)
        return english_ratio >= 0.3
    
    return False

def advanced_clean_text(text):
    """
    Advanced text cleaning with more comprehensive rules
    """
    if not isinstance(text, str):
        return ""
    
    # Handle contractions first
    text = handle_contractions(text)
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags and entities
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&\w+;', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Remove numbers (optional, can be commented out if numbers are important)
    # text = re.sub(r'\d+', '', text)
    
    # Keep only letters, spaces, and basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Normalize whitespace
    text = remove_duplicate_spaces(text)
    
    return text

def get_text_statistics(texts):
    """
    Get comprehensive statistics about a collection of texts
    
    Args:
        texts (list): List of texts to analyze
        
    Returns:
        dict: Dictionary containing various statistics
    """
    if not texts:
        return {}
    
    # Filter out empty texts
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    
    if not valid_texts:
        return {"error": "No valid texts found"}
    
    # Calculate statistics
    char_lengths = [len(text) for text in valid_texts]
    word_counts = [len(text.split()) for text in valid_texts]
    
    # Word length statistics
    word_lengths = []
    for text in valid_texts:
        words = text.split()
        word_lengths.extend([len(word) for word in words])
    
    stats = {
        'total_texts': len(valid_texts),
        'total_characters': sum(char_lengths),
        'total_words': sum(word_counts),
        'avg_chars_per_text': np.mean(char_lengths),
        'avg_words_per_text': np.mean(word_counts),
        'avg_chars_per_word': np.mean(word_lengths) if word_lengths else 0,
        'min_chars_per_text': min(char_lengths),
        'max_chars_per_text': max(char_lengths),
        'min_words_per_text': min(word_counts),
        'max_words_per_text': max(word_counts),
        'std_chars_per_text': np.std(char_lengths),
        'std_words_per_text': np.std(word_counts)
    }
    
    return stats

def preprocess_with_options(text, options=None):
    """
    Preprocess text with customizable options
    
    Args:
        text (str): Text to preprocess
        options (dict): Preprocessing options
        
    Returns:
        str: Preprocessed text
    """
    if options is None:
        options = {
            'lowercase': True,
            'remove_urls': True,
            'remove_html': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'expand_contractions': True,
            'remove_stopwords': True,
            'apply_stemming': True,
            'min_word_length': 2
        }
    
    if not isinstance(text, str):
        return ""
    
    processed_text = text
    
    # Apply options in order
    if options.get('expand_contractions', True):
        processed_text = handle_contractions(processed_text)
    
    if options.get('lowercase', True):
        processed_text = processed_text.lower()
    
    if options.get('remove_urls', True):
        processed_text = re.sub(r'http\S+|www\S+|https\S+', '', processed_text)
    
    if options.get('remove_html', True):
        processed_text = re.sub(r'<.*?>', '', processed_text)
    
    if options.get('remove_numbers', False):
        processed_text = re.sub(r'\d+', '', processed_text)
    
    if options.get('remove_punctuation', True):
        processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
    
    # Normalize whitespace
    processed_text = remove_duplicate_spaces(processed_text)
    
    if options.get('remove_stopwords', True):
        words = processed_text.split()
        min_length = options.get('min_word_length', 2)
        words = [word for word in words if word not in stop_words and len(word) > min_length]
        processed_text = ' '.join(words)
    
    if options.get('apply_stemming', True):
        words = processed_text.split()
        words = [stemmer.stem(word) for word in words]
        processed_text = ' '.join(words)
    
    return processed_text.strip()

# Main function for testing
def main():
    """
    Test the preprocessing functions
    """
    sample_texts = [
        "This is a GREAT article about fake news detection! Check out https://example.com for more info.",
        "I can't believe this news story. It's absolutely SHOCKING!!! The government doesn't want you to know this...",
        "According to researchers at Stanford University, the new study published in Nature shows promising results.",
        "BREAKING: Local man discovers the SECRET to eternal youth that doctors HATE! Click here now!!!"
    ]
    
    print("Text Preprocessing Test")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}:")
        print(f"Original: {text}")
        processed = preprocess_text(text)
        print(f"Processed: {processed}")
        
        validation = validate_preprocessing(text, processed)
        print(f"Validation: {validation}")
    
    # Test statistics
    print(f"\nDataset Statistics:")
    stats = get_text_statistics(sample_texts)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()