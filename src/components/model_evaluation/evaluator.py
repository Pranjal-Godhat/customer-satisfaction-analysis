import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
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

def generate_classification_report(y_test: np.ndarray, y_pred: np.ndarray, average_method: str = 'weighted') -> str:
    try:
        try:
            report = classification_report(y_test, y_pred, average=average_method, zero_division=0)
        except TypeError:
            report = classification_report(y_test, y_pred, average=average_method)
        logger.info('Classification report generated')
        return report
    except Exception as e:
        try:
            report = classification_report(y_test, y_pred)
            logger.info('Classification report generated (fallback)')
            return report
        except Exception as e2:
            logger.error('Error generating classification report: %s', e2)
            raise e2

def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray, average_method: str = 'weighted') -> Dict[str, float]:
    try:
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}
        for metric_name, metric_func in [("precision", precision_score), ("recall", recall_score), ("f1_score", f1_score)]:
            try:
                metrics[metric_name] = metric_func(y_test, y_pred, average=average_method)
            except TypeError:
                try:
                    metrics[metric_name] = metric_func(y_test, y_pred)
                except Exception:
                    metrics[metric_name] = 0.0
        logger.info('Metrics calculated: %s', metrics)
        return metrics
    except Exception as e:
        logger.error('Error calculating metrics: %s', e)
        raise

def plot_roc_curves(models: Dict, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder, save_dir: str = './data_bucket'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                for class_idx in range(y_prob.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, class_idx], pos_label=class_idx)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{name} (Class {label_encoder.inverse_transform([class_idx])[0]} - AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
        plt.close()
        logger.info('ROC curves saved to %s', save_dir)
    except Exception as e:
        logger.error('Error plotting ROC curves: %s', e)
        raise

def plot_metrics_bar_chart(results_df: pd.DataFrame, save_dir: str = './data_bucket'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        metrics_cols = ['Testing Accuracy', 'Precision', 'Recall', 'F1 Score']
        results_df.set_index('Model')[metrics_cols].plot(kind='bar', figsize=(12, 6))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close()
        logger.info('Model comparison chart saved to %s', save_dir)
    except Exception as e:
        logger.error('Error plotting metrics: %s', e)
        raise

def main():
    try:
        logger.info('Starting model evaluation pipeline')
        params = load_params()
        logger.info('Model evaluation pipeline completed')
    except Exception as e:
        logger.error('Failed to complete model evaluation pipeline: %s', e)
        raise

if __name__ == '__main__':
    main()
