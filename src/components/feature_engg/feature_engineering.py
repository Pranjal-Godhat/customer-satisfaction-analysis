import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, Any

from src.logger import logger

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise

def feature_engineering(df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
    try:
        df = df.copy()
        threshold = params['feature_engineering']['top_categories_threshold'] if params else 3000

        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

        df['day_of_week_int'] = df['order_purchase_timestamp'].dt.weekday + 1
        df['hour'] = df['order_purchase_timestamp'].dt.hour
        df['month'] = df['order_purchase_timestamp'].dt.month
        df['year'] = df['order_purchase_timestamp'].dt.year
        df['date'] = df['order_purchase_timestamp'].dt.to_period('M')
        df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

        df.rename(columns={'product_name_lenght': 'product_name_length'}, inplace=True)

        category_counts = df['product_category_name'].value_counts()
        common_categories = category_counts[category_counts >= threshold].index
        df['simplified_category'] = df['product_category_name'].where(df['product_category_name'].isin(common_categories), 'other')

        df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M')

        logger.info('Feature engineering completed. Shape: %s', df.shape)
        return df
    except Exception as e:
        logger.error('Error in feature engineering: %s', e)
        raise

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.dropna(subset=['review_comment_message', 'review_comment_title', 'product_category_name']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['review_comment_message']).reset_index(drop=True)
        logger.info('Data cleaning completed. Shape: %s', df.shape)
        return df
    except Exception as e:
        logger.error('Error in data cleaning: %s', e)
        raise

def main():
    try:
        logger.info('Starting feature engineering pipeline')
        params = load_params()
        logger.info('Feature engineering pipeline completed')
    except Exception as e:
        logger.error('Failed to complete feature engineering pipeline: %s', e)
        raise

if __name__ == '__main__':
    main()
