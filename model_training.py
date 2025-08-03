import os
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Load the fake news dataset
    Expected columns: 'title', 'text', 'label' (0 for real, 1 for fake)
    """
    try:
        if not filepath.endswith('.csv'):
            raise ValueError("Unsupported file format. Please use CSV.")
            
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Define column mappings
        column_mappings = {
            'text': ['text', 'content', 'article', 'news'],
            'label': ['label', 'class', 'target', 'fake'],
            'title': ['title', 'headline', 'subject']
        }
        
        # Find columns
        found_columns = {}
        for col_type, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    found_columns[col_type] = col
                    break
        
        if 'text' not in found_columns or 'label' not in found_columns:
            raise ValueError(f"Required columns missing. Available columns: {list(df.columns)}")
        
        # Combine title and text if available
        text_column = found_columns['text']
        if 'title' in found_columns:
            df['combined_text'] = df[found_columns['title']].fillna('') + ' ' + df[found_columns['text']].fillna('')
            text_column = 'combined_text'
        
        # Clean data
        df = df.dropna(subset=[text_column, found_columns['label']])
        
        # Standardize labels to binary (0 for real, 1 for fake)
        label_maps = {
            frozenset(['REAL', 'FAKE']): {'REAL': 0, 'FAKE': 1},
            frozenset(['real', 'fake']): {'real': 0, 'fake': 1},
            frozenset([True, False]): {True: 1, False: 0}
        }
        
        unique_labels = set(df[found_columns['label']].unique())
        if len(unique_labels) != 2:
            raise ValueError(f"Expected binary labels, got: {unique_labels}")
        
        if unique_labels != {0, 1}:
            for label_set, mapping in label_maps.items():
                if unique_labels == set(label_set):
                    df[found_columns['label']] = df[found_columns['label']].map(mapping)
                    break
            else:
                # If no predefined mapping found, map first value to 0 and second to 1
                unique_labels = list(unique_labels)
                df[found_columns['label']] = df[found_columns['label']].map({unique_labels[0]: 0, unique_labels[1]: 1})
        
        print(f"Label distribution:\n{df[found_columns['label']].value_counts()}")
        return df[text_column].values, df[found_columns['label']].values
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_sample_data():
    """
    Create balanced sample dataset for demonstration
    """
    print("Creating sample dataset for demonstration...")
    
    samples = {
        'fake': [
            "BREAKING: Scientists discover that drinking water turns people into fish! Government coverup exposed!",
            "SHOCKING: Local man claims he can fly after eating special berries found in his backyard!",
            "EXCLUSIVE: Secret government program uses mind control through television broadcasts!",
            "UNBELIEVABLE: Doctors hate this one simple trick that cures everything!",
            "EXPOSED: The truth about how aliens built the pyramids that THEY don't want you to know!"
        ],
        'real': [
            "The Federal Reserve announced today that interest rates will remain unchanged following their monthly meeting.",
            "Researchers at Harvard University published a new study on climate change impacts in the Journal of Environmental Science.",
            "The city council approved the new budget proposal in a 7-3 vote yesterday evening.",
            "Stock markets closed higher today as technology companies reported better-than-expected quarterly earnings.",
            "The World Health Organization released new guidelines for vaccination schedules based on clinical trials."
        ]
    }
    
    # Create balanced dataset
    texts = samples['fake'] + samples['real']
    labels = [1] * len(samples['fake']) + [0] * len(samples['real'])
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return np.array(texts), np.array(labels)

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the fake news detection model with optimized pipeline"""
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create and fit pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            class_weight='balanced'
        ))
    ])
    
    # Preprocess and train
    X_train_processed = [preprocess_text(text) for text in X_train]
    X_test_processed = [preprocess_text(text) for text in X_test]
    
    # Filter empty texts
    valid_train = [i for i, text in enumerate(X_train_processed) if text.strip()]
    valid_test = [i for i, text in enumerate(X_test_processed) if text.strip()]
    
    X_train_clean = [X_train_processed[i] for i in valid_train]
    y_train_clean = y_train[valid_train]
    X_test_clean = [X_test_processed[i] for i in valid_test]
    y_test_clean = y_test[valid_test]
    
    # Fit pipeline
    pipeline.fit(X_train_clean, y_train_clean)
    
    # Evaluate
    y_pred = pipeline.predict(X_test_clean)
    accuracy = accuracy_score(y_test_clean, y_pred)
    
    # Print metrics
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test_clean, y_pred, target_names=['Real News', 'Fake News']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_clean, y_pred)
    print(cm)
    print("[[True Negatives, False Positives]")
    print(" [False Negatives, True Positives]]")
    
    return pipeline['classifier'], pipeline['vectorizer'], accuracy

def save_model(model, vectorizer, model_dir='models'):
    """
    Save the trained model and vectorizer if accuracy meets threshold
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        model_files = {
            'model': ('fake_news_model.pkl', model),
            'vectorizer': ('tfidf_vectorizer.pkl', vectorizer)
        }
        
        for component, (filename, obj) in model_files.items():
            filepath = os.path.join(model_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            print(f"{component.title()} saved to {filepath}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def main(data_path=None, model_dir='models', min_accuracy=0.7):
    """
    Main function to run the fake news detection training pipeline
    
    Args:
        data_path (str, optional): Path to training data CSV
        model_dir (str): Directory to save model files
        min_accuracy (float): Minimum accuracy threshold for saving model
    """
    try:
        print("Fake News Detection Model Training")
        print("=" * 50)
        
        # Load or create training data
        if data_path and os.path.exists(data_path):
            X, y = load_data(data_path)
            print("Dataset loaded successfully!")
        else:
            if data_path:
                print(f"Warning: Data path {data_path} not found. Using sample data.")
            X, y = create_sample_data()
        
        # Train and evaluate model
        model, vectorizer, accuracy = train_model(X, y)
        
        # Save model if accuracy meets threshold
        if accuracy >= min_accuracy:
            save_model(model, vectorizer, model_dir)
            print(f"\nModel training completed successfully! Accuracy: {accuracy:.2f}")
        else:
            print(f"\nModel accuracy ({accuracy:.2f}) below threshold ({min_accuracy}). Model not saved.")
            
        print("You can now run the Flask app with: python app.py")
            
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()