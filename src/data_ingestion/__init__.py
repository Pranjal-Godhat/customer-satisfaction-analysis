import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging
from src.logger import logging

dataset_paths = {'customers' : './data/olist_customers_dataset.csv',
        "geolocation" : "./data/olist_geolocation_dataset.csv",
        "order_items" : "./data/olist_order_items_dataset.csv",
        "order_payments" : './data/olist_order_payments_dataset.csv',
        "order_reviews" : './data/olist_order_reviews_dataset.csv',
        "orders" : './data/olist_orders_dataset.csv',
        "products" : './data/olist_products_dataset.csv',
        "sellers" : './data/olist_sellers_dataset.csv',
        "product_category" : './data/product_category_name_translation.csv'}


def load_params(params_path : str ) -> dict:
    # load params fom yaml file
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('params retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('file not found %s', params_path)
        raise
    except yaml.YamlError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_dataset(dataset_paths : dict) -> pd.DataFrame :
     # load .csv file as dataframe

    try:
        data = {}
        for key in dataset_paths:
            data[key]= pd.read_csv(dataset_paths[key])

        #merging all data in one df
        df = data['orders'].merge(data['order_items'], on='order_id', how='inner')
        df = df.merge(data['order_payments'], on='order_id', how='inner')
        df = df.merge(data['order_reviews'], on='order_id', how='inner')
        df = df.merge(data['products'], on='product_id', how='inner')
        df = df.merge(data['customers'], on='customer_id', how='inner')
        df = df.merge(data['sellers'], on='seller_id', how='inner')

        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logging.error('unexpected error occured in loading data: %s', e)
        raise

def save_data(train_data: pd.DataFrame , test_data: pd.DataFrame, data_path: str) -> None:
    #save train and test data in data_bucket

    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occured while saving data %s', e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data ingestion']['test_size']

        df = load_dataset(dataset_paths)

        train_data, test_data = train_test_split(df , test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data_bucket')

    except Exception as e:
        logging.error('Failed to complete data ingestion process: %s', e)

if __name__ == '__main__':
    main()

