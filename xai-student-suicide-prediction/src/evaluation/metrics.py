"""
Evaluation Metrics for Student Suicide Risk Prediction

This module provides comprehensive evaluation metrics including:
- AUC-ROC, AUC-PR, Precision, Recall, F1-score
- Confusion matrix analysis
- Threshold optimization
- Performance comparison between models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  threshold: float = 0.5) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model performance.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Convert to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Basic classification metrics
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }

        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        logger.info(f"Model evaluation completed. AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Dict[str, Any]:
    """
    Find optimal classification threshold for a given metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'specificity')

    Returns:
        Dictionary with optimal threshold and metrics
    """
    try:
        thresholds = np.arange(0.1, 0.9, 0.01)
        metric_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'specificity':
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp = cm[0]
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    score = 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            metric_scores.append(score)

        # Find optimal threshold
        optimal_idx = np.argmax(metric_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = metric_scores[optimal_idx]

        # Evaluate at optimal threshold
        optimal_metrics = evaluate_model(y_true, y_pred_proba, optimal_threshold)

        result = {
            'optimal_threshold': optimal_threshold,
            'optimal_score': optimal_score,
            'metric_optimized': metric,
            'threshold_range': [thresholds[0], thresholds[-1]],
            'all_thresholds': thresholds.tolist(),
            'all_scores': metric_scores,
            'optimal_metrics': optimal_metrics
        }

        logger.info(f"Optimal threshold found: {optimal_threshold:.3f} (max {metric}: {optimal_score:.4f})")
        return result

    except Exception as e:
        logger.error(f"Error finding optimal threshold: {e}")
        raise

def compare_models(y_true: np.ndarray, model_predictions: Dict[str, np.ndarray],
                  thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Compare multiple models on the same test set.

    Args:
        y_true: True labels
        model_predictions: Dictionary of model_name -> predicted probabilities
        thresholds: Optional dictionary of model_name -> threshold

    Returns:
        Dictionary with comparison results
    """
    try:
        comparison_results = {}
        model_metrics = {}

        for model_name, y_pred_proba in model_predictions.items():
            threshold = thresholds.get(model_name, 0.5) if thresholds else 0.5
            metrics = evaluate_model(y_true, y_pred_proba, threshold)
            model_metrics[model_name] = metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            model_name: {
                'auc_roc': metrics['auc_roc'],
                'auc_pr': metrics['auc_pr'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            }
            for model_name, metrics in model_metrics.items()
        }).T

        # Find best model for each metric
        best_models = {}
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_value = comparison_df[metric].max()
            best_models[metric] = {'model': best_model, 'value': best_value}

        comparison_results = {
            'model_metrics': model_metrics,
            'comparison_table': comparison_df.to_dict(),
            'best_models': best_models,
            'ranking': comparison_df.rank(ascending=False).mean(axis=1).sort_values().to_dict()
        }

        logger.info("Model comparison completed")
        return comparison_results

    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise

def plot_evaluation_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          model_name: str = "Model", save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve and Precision-Recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{model_name} - ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{model_name} - Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation curves saved to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting evaluation curves: {e}")
        raise

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model", save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])

        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise

def calculate_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate calibration metrics (reliability curve).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    try:
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        observed_probs = []
        predicted_probs = []

        for i in range(n_bins):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i + 1])
            if mask.sum() > 0:
                observed_prob = y_true[mask].mean()
                predicted_prob = y_pred_proba[mask].mean()

                observed_probs.append(observed_prob)
                predicted_probs.append(predicted_prob)

        # Calculate calibration metrics
        observed_probs = np.array(observed_probs)
        predicted_probs = np.array(predicted_probs)

        # Expected Calibration Error (ECE)
        ece = np.abs(observed_probs - predicted_probs).mean()

        # Maximum Calibration Error (MCE)
        mce = np.abs(observed_probs - predicted_probs).max()

        calibration_results = {
            'ece': ece,
            'mce': mce,
            'bin_centers': bin_centers.tolist(),
            'observed_probs': observed_probs.tolist(),
            'predicted_probs': predicted_probs.tolist(),
            'n_bins': n_bins
        }

        logger.info(f"Calibration metrics calculated. ECE: {ece:.4f}, MCE: {mce:.4f}")
        return calibration_results

    except Exception as e:
        logger.error(f"Error calculating calibration metrics: {e}")
        raise

def generate_evaluation_report(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             model_name: str = "Model", threshold: float = 0.5,
                             output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with plots and metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        threshold: Classification threshold
        output_dir: Directory to save plots (optional)

    Returns:
        Dictionary with complete evaluation report
    """
    try:
        logger.info(f"Generating evaluation report for {model_name}...")

        # Basic evaluation
        metrics = evaluate_model(y_true, y_pred_proba, threshold)

        # Optimal threshold analysis
        optimal_threshold_analysis = find_optimal_threshold(y_true, y_pred_proba, 'f1')

        # Calibration analysis
        calibration = calculate_calibration_metrics(y_true, y_pred_proba)

        # Generate plots if output directory provided
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Evaluation curves
            curves_path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_curves.png"
            plot_evaluation_curves(y_true, y_pred_proba, model_name, str(curves_path))

            # Confusion matrix
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm_path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            plot_confusion_matrix(y_true, y_pred, model_name, str(cm_path))

        report = {
            'model_name': model_name,
            'metrics': metrics,
            'optimal_threshold': optimal_threshold_analysis,
            'calibration': calibration,
            'plots_generated': output_dir is not None,
            'evaluation_summary': {
                'meets_auc_target': metrics['auc_roc'] >= 0.85,
                'meets_recall_target': metrics['recall'] >= 0.75,
                'meets_precision_target': metrics['precision'] >= 0.60,
                'meets_f1_target': metrics['f1'] >= 0.70
            }
        }

        logger.info(f"Evaluation report generated for {model_name}")
        return report

    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")
        raise
