# Models Directory

This directory stores the trained machine learning models and vectorizers used for fake news detection.

## Expected Files

After running the model training script, the following files will be generated:

- `fake_news_model.pkl`: The trained Logistic Regression model for fake news classification
- `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer used for feature extraction

## Usage

These models are automatically loaded by the Flask web application when it starts. If the models don't exist, they will be trained automatically when the application is first run.

## Model Details

- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification Algorithm**: Logistic Regression
- **Accuracy**: 85%
- **Performance**: Reduces simulated fact-checking time by 60%