import os
import pandas as pd
import kaggle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess_data, prepare_features

def download_dataset():
    # Check if dataset exists
    if not (os.path.exists('data/Fake.csv') and os.path.exists('data/True.csv')):
        print('Downloading dataset from Kaggle...')
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'clmentbisaillon/fake-and-real-news-dataset',
                path='data',
                unzip=True
            )
            print('Dataset downloaded successfully')
        except Exception as e:
            print(f'Error downloading dataset: {e}')
            print('Please ensure:')
            print('1. kaggle.json credentials file exists in ~/.kaggle/')
            print('2. You accepted competition rules on Kaggle website')
            exit(1)

def train_model():
    # Download dataset first
    download_dataset()
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data/Fake.csv', 'data/True.csv')
    
    print("Preparing features...")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = prepare_features(df)
    
    print("Training model...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_model()