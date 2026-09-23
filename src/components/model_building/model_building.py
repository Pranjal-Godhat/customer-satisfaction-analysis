import pandas as pd
import logging
import yaml
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def initialize_models(params: Dict[str, Any]) -> Dict:
    try:
        model_params = params['model_training']
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        models = {
            "GaussianNB": GaussianNB(),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=model_params['random_state']),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=model_params['n_estimators'], random_state=model_params['random_state']),
            "LogisticRegression": LogisticRegression(random_state=50, max_iter=model_params['max_iter']),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=45),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=model_params['n_neighbors']),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=model_params['random_state']),
            "SVC": SVC(kernel='linear', random_state=model_params['random_state'], probability=True),
        }
        logger.info('Initialized %d models', len(models))
        return models
    except Exception as e:
        logger.error('Error initializing models: %s', e)
        raise

def train_models(models: Dict, X_train, y_train) -> Dict:
    try:
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            logger.info('Model %s trained. Training accuracy: %.4f', name, train_accuracy)
            trained_models[name] = model
        return trained_models
    except Exception as e:
        logger.error('Error training models: %s', e)
        raise

def evaluate_models(models: Dict, X_test, y_test, label_encoder) -> pd.DataFrame:
    try:
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Training Accuracy": accuracy_score(y_test, model.predict(X_test)),
                "Testing Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            })
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Testing Accuracy', ascending=False)
        logger.info('Model evaluation completed')
        return results_df
    except Exception as e:
        logger.error('Error evaluating models: %s', e)
        raise

def main():
    try:
        logger.info('Starting model building pipeline')
        params = load_params()
        logger.info('Model building pipeline completed')
    except Exception as e:
        logger.error('Failed to complete model building pipeline: %s', e)
        raise

if __name__ == '__main__':
    main()
