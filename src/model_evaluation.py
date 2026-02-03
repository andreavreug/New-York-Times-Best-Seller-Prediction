"""
Model evaluation module for assessing classification performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)


class ModelEvaluator:
    """Evaluate classification model performance."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        pass
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         average: str = 'binary') -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: List[str] = None,
                             save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 target_names: List[str] = None) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate ROC-AUC score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            ROC-AUC score
        """
        return roc_auc_score(y_true, y_prob)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                       model_name: str = 'Model',
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    model_name: str = 'Model',
                                    save_path: str = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'{model_name} (AP = {avg_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame comparing models
        """
        df = pd.DataFrame(results).T
        df = df.sort_values('f1', ascending=False)
        return df
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'Model') -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with all evaluation results
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Get confusion matrix
        cm = self.get_confusion_matrix(y_test, y_pred)
        
        # Get classification report
        report = self.get_classification_report(y_test, y_pred,
                                               target_names=['Not Best Seller', 'Best Seller'])
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Add ROC-AUC if probabilities available
        if y_prob is not None:
            results['roc_auc'] = self.calculate_roc_auc(y_test, y_prob)
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """
        Print evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Model: {results['model_name']}")
        print(f"{'='*60}")
        
        print("\nMetrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        if 'roc_auc' in results:
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        print("\nClassification Report:")
        print(results['classification_report'])
