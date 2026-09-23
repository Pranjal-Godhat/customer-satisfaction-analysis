import pandas as pd
import numpy as np
import string
import logging
import yaml
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from src.logger import logger

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading params: %s', e)
        raise

def download_nltk_data() -> None:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        logger.info('NLTK data downloaded successfully')
    except Exception as e:
        logger.error('Failed to download NLTK data: %s', e)
        raise

def get_portuguese_stopwords() -> set:
    try:
        return set(stopwords.words('portuguese'))
    except Exception as e:
        logger.error('Error loading Portuguese stopwords: %s', e)
        raise

def clean_and_tokenize(text: str, stop_words: set) -> Tuple[str, List[str]]:
    try:
        if not isinstance(text, str):
            return "", []
        cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = cleaned_text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words), filtered_words
    except Exception as e:
        logger.error('Error in text cleaning: %s', e)
        raise

def preprocess_nlp_df(df: pd.DataFrame, stop_words: set) -> pd.DataFrame:
    try:
        df = df.copy()
        df = df.dropna(subset=['review_comment_message']).reset_index(drop=True)
        df[['review_comment_message_clean', 'review_comment_message_tokens']] = df['review_comment_message'].apply(
            lambda text: pd.Series(clean_and_tokenize(text, stop_words))
        )
        df.dropna(subset=['review_comment_title', 'review_comment_message'], inplace=True)
        df.drop_duplicates(subset=['review_comment_message', 'review_comment_title'], inplace=True)
        logger.info('NLP preprocessing completed. Shape: %s', df.shape)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error('Error in NLP preprocessing: %s', e)
        raise

def classify_sentiment(df: pd.DataFrame, column_name: str = 'review_comment_message_clean',
                       positive_threshold: float = 0.05, negative_threshold: float = -0.05) -> pd.DataFrame:
    try:
        analyzer = SentimentIntensityAnalyzer()
        def get_sentiment_classification(text: str) -> str:
            scores = analyzer.polarity_scores(text)
            if scores['compound'] >= positive_threshold:
                return 'Positive'
            elif scores['compound'] <= negative_threshold:
                return 'Negative'
            else:
                return 'Neutral'
        df = df.copy()
        df[f'{column_name}_sentiment'] = df[column_name].map(get_sentiment_classification)
        logger.info('Sentiment classification completed')
        return df
    except Exception as e:
        logger.error('Error in sentiment classification: %s', e)
        raise

def prepare_training_data(df: pd.DataFrame, max_features: int = 5000,
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    try:
        X = df['review_comment_message']
        y = df['review_comment_message_clean_sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        X_train_dense = X_train_tfidf.toarray()
        X_test_dense = X_test_tfidf.toarray()

        logger.info('Training data prepared. Train shape: %s, Test shape: %s',
                    X_train_dense.shape, X_test_dense.shape)

        return X_train_dense, X_test_dense, y_train_encoded, y_test_encoded, label_encoder, vectorizer
    except Exception as e:
        logger.error('Error in training data preparation: %s', e)
        raise

def main() -> None:
    try:
        logger.info('Starting data preprocessing pipeline')
        params = load_params()
        prep_params = params['data_preprocessing']
        download_nltk_data()
        stop_words = get_portuguese_stopwords()
        logger.info('Data preprocessing pipeline completed successfully')
    except Exception as e:
        logger.error('Failed to complete data preprocessing pipeline: %s', e)
        raise

if __name__ == '__main__':
    main()
