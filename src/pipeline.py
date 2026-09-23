import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logger
from src.components.data_ingestion.data_ingestion import load_params, feature_engineering, clean_and_prepare, save_data
from src.components.data_preprocessing.data_preprocessing import download_nltk_data, get_portuguese_stopwords, preprocess_nlp_df, classify_sentiment, prepare_training_data
from src.model_training.trainer import initialize_models, train_models, evaluate_models, plot_confusion_matrices, plot_feature_importance
from src.model_evaluation.evaluator import generate_classification_report, calculate_metrics, plot_roc_curves, plot_metrics_bar_chart
from sklearn.model_selection import train_test_split

def create_500_row_dataset() -> pd.DataFrame:
    try:
        dataset_paths = {
            'customers': './data/olist_customers_dataset.csv',
            'geolocation': './data/olist_geolocation_dataset.csv',
            'order_items': './data/olist_order_items_dataset.csv',
            'order_payments': './data/olist_order_payments_dataset.csv',
            'order_reviews': './data/olist_order_reviews_dataset.csv',
            'orders': './data/olist_orders_dataset.csv',
            'products': './data/olist_products_dataset.csv',
            'sellers': './data/olist_sellers_dataset.csv',
            'product_category': './data/product_category_name_translation.csv'
        }

        data = {}
        for key in dataset_paths:
            df = pd.read_csv(dataset_paths[key])
            data[key] = df
            logger.info('Loaded %s with %d rows', key, len(df))

        df = data['orders'].merge(data['order_items'], on='order_id', how='inner')
        df = df.merge(data['order_payments'], on='order_id', how='inner')
        df = df.merge(data['order_reviews'], on='order_id', how='inner')
        df = df.merge(data['products'], on='product_id', how='inner')
        df = df.merge(data['customers'], on='customer_id', how='inner')
        df = df.merge(data['sellers'], on='seller_id', how='inner')

        if len(df) > 500:
            df = df.head(500)
            logger.info('Trimmed to 500 rows')

        logger.info('Merged dataset shape: %s', df.shape)
        return df
    except Exception as e:
        logger.error('Error creating 500-row dataset: %s', e)
        raise

def run_pipeline():
    try:
        logger.info('='*60)
        logger.info('Starting MLOps Pipeline')
        logger.info('='*60)

        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        max_features = params['data_preprocessing']['max_features_tfidf']
        sentiment_pos = params['data_preprocessing']['sentiment_threshold_positive']
        sentiment_neg = params['data_preprocessing']['sentiment_threshold_negative']

        logger.info('Step 1: Data Ingestion')
        df = create_500_row_dataset()

        logger.info('Step 2: Feature Engineering')
        df = feature_engineering(df, params)
        df = clean_and_prepare(df)
        logger.info('Dataset shape after cleaning: %s', df.shape)

        logger.info('Step 3: Data Preprocessing - NLP')
        download_nltk_data()
        stop_words = get_portuguese_stopwords()
        df = preprocess_nlp_df(df, stop_words)
        df = classify_sentiment(df, 'review_comment_message_clean', positive_threshold=sentiment_pos, negative_threshold=sentiment_neg)
        logger.info('Dataset shape after NLP preprocessing: %s', df.shape)

        logger.info('Step 4: Prepare Training Data')
        X_train_dense, X_test_dense, y_train_encoded, y_test_encoded, label_encoder, vectorizer = prepare_training_data(
            df, max_features=max_features, test_size=test_size, random_state=random_state
        )
        logger.info('Training data shapes - X_train: %s, X_test: %s', X_train_dense.shape, X_test_dense.shape)

        logger.info('Step 5: Model Training')
        models = initialize_models(params)
        trained_models = train_models(models, X_train_dense, y_train_encoded)

        logger.info('Step 6: Model Evaluation')
        results_df = evaluate_models(trained_models, X_test_dense, y_test_encoded, label_encoder)
        print(results_df.to_string())

        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]

        logger.info('Step 7: Generate Reports')
        y_pred = best_model.predict(X_test_dense)
        report = generate_classification_report(y_test_encoded, y_pred)
        logger.info('Classification Report:\n%s', report)

        metrics = calculate_metrics(y_test_encoded, y_pred)
        logger.info('Best Model Metrics: %s', metrics)

        plot_confusion_matrices(trained_models, X_test_dense, y_test_encoded, label_encoder)
        plot_feature_importance(trained_models, vectorizer.get_feature_names_out())
        plot_roc_curves(trained_models, X_test_dense, y_test_encoded, label_encoder)
        plot_metrics_bar_chart(results_df)

        logger.info('Step 8: Save Results')
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, data_path='./data_bucket')
        results_df.to_csv('./data_bucket/results.csv', index=False)

        logger.info('='*60)
        logger.info('MLOps Pipeline Completed Successfully!')
        logger.info('='*60)
        return results_df

    except Exception as e:
        logger.error('Pipeline failed: %s', e)
        raise

if __name__ == '__main__':
    run_pipeline()
