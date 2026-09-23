import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from src.logger import logger

dataset_paths = {'customers' : './data/olist_customers_dataset.csv',
        "geolocation" : "./data/olist_geolocation_dataset.csv",
        "order_items" : "./data/olist_order_items_dataset.csv",
        "order_payments" : './data/olist_order_payments_dataset.csv',
        "order_reviews" : './data/olist_order_reviews_dataset.csv',
        "orders" : './data/olist_orders_dataset.csv',
        "products" : './data/olist_products_dataset.csv',
        "sellers" : './data/olist_sellers_dataset.csv',
        "product_category" : './data/product_category_name_translation.csv'}

def load_params(params_path: str = 'params.yaml') -> dict:
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
        logger.error('Unexpected error: %s', e)
        raise

def load_dataset(dataset_paths: dict) -> pd.DataFrame:
    try:
        data = {}
        for key in dataset_paths:
            data[key] = pd.read_csv(dataset_paths[key])
        logger.info('Loaded %d datasets', len(data))

        df = data['orders'].merge(data['order_items'], on='order_id', how='inner')
        df = df.merge(data['order_payments'], on='order_id', how='inner')
        df = df.merge(data['order_reviews'], on='order_id', how='inner')
        df = df.merge(data['products'], on='product_id', how='inner')
        df = df.merge(data['customers'], on='customer_id', how='inner')
        df = df.merge(data['sellers'], on='seller_id', how='inner')

        logger.info('Merged dataset shape: %s', df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error in loading data: %s', e)
        raise

def feature_engineering(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str = './data_bucket') -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving data: %s', e)
        raise

def main():
    try:
        logger.info('Starting data ingestion pipeline')
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']

        df = load_dataset(dataset_paths)

        df = feature_engineering(df, params)
        df = clean_and_prepare(df)

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

        save_data(train_data, test_data, data_path='./data_bucket')

        logger.info('Data ingestion pipeline completed successfully')
        logger.info('Train shape: %s, Test shape: %s', train_data.shape, test_data.shape)
    except Exception as e:
        logger.error('Failed to complete data ingestion process: %s', e)
        raise

if __name__ == '__main__':
    main()
