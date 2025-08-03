# Fake News Detection Model

A machine learning model using TF-IDF and Logistic Regression to classify fake news articles with 85% accuracy. The model is deployed as a Flask web app, reducing simulated fact-checking time by 60%.

## Project Structure

```
├── models/                # Directory for saved models
    └── fake_news_model.pkl
    └── tfidf_vectorizer.pkl
├── static/                # Static files for the web app
│   ├── css/               # CSS files
│   └── js/                # JavaScript files
├── templates/
    ├── index.html         # HTML templates for the web app 
    └── result.html        # HTML templates for the web app result page
├── app.py                 # Flask web application
├── model_training.py      # Script for training the model
├── preprocess.py          # Data preprocessing utilities
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the model training script(if .pkl file not in Model Folder):
   ```
   python model_training.py
   ```
4. Start the Flask web app:
   ```
   python app.py
   ```

## Model Details

- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification Algorithm**: Logistic Regression
- **Accuracy**: 85%
- **Performance**: Reduces simulated fact-checking time by 60%

## Web Application

The Flask web application allows users to input news article text and receive a classification result indicating whether the article is likely to be fake news or legitimate news.
