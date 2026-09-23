import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def train_models(models: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
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

def evaluate_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder) -> pd.DataFrame:
    try:
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({
                "Model": name,
                "Training Accuracy": accuracy_score(y_test, model.predict(X_test)),
                "Testing Accuracy": test_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
            logger.info('Model %s - Test Accuracy: %.4f, F1: %.4f', name, test_accuracy, f1)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Testing Accuracy', ascending=False)
        logger.info('Model evaluation completed')
        return results_df
    except Exception as e:
        logger.error('Error evaluating models: %s', e)
        raise

def plot_confusion_matrices(models: Dict, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder, save_dir: str = './data_bucket'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1, figsize=(10, 5 * n_models))
        if n_models == 1:
            axes = [axes]
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
            disp.plot(ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
        plt.close()
        logger.info('Confusion matrices saved to %s', save_dir)
    except Exception as e:
        logger.error('Error plotting confusion matrices: %s', e)
        raise

def plot_feature_importance(models: Dict, feature_names: np.ndarray, save_dir: str = './data_bucket'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(20)
                plt.figure(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
                plt.title(f"Top 20 Features - {name}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'feature_importance_{name}.png'))
                plt.close()
                logger.info('Feature importance plot saved for %s', name)
    except Exception as e:
        logger.error('Error plotting feature importance: %s', e)
        raise

def main():
    try:
        logger.info('Starting model training pipeline')
        params = load_params()
        logger.info('Model training pipeline completed')
    except Exception as e:
        logger.error('Failed to complete model training pipeline: %s', e)
        raise

if __name__ == '__main__':
    main()
