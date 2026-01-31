"""
SHAP Explainer for Student Suicide Risk Prediction

This module provides SHAP-based explainability for both global and local
interpretations of Random Forest and XGBoost models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP-based explainer for suicide risk prediction models.

    Provides both global and local explanations using SHAP values.
    """

    def __init__(self, model: Any, X_background: pd.DataFrame, model_type: str = 'tree'):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (Random Forest or XGBoost)
            X_background: Background dataset for SHAP explanations
            model_type: Type of model ('tree' for RF/XGB, 'linear' for linear models)
        """
        self.model = model
        self.X_background = X_background
        self.model_type = model_type
        self.feature_names = X_background.columns.tolist()

        # Initialize SHAP explainer
        try:
            if model_type == 'tree':
                self.explainer = shap.TreeExplainer(model, X_background)
            elif model_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_background)
            else:
                self.explainer = shap.Explainer(model, X_background)

            logger.info(f"SHAP explainer initialized for {model_type} model")

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            raise

    def explain_global(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate global SHAP explanations for the entire test set.

        Args:
            X_test: Test features

        Returns:
            Dictionary with global SHAP analysis results
        """
        try:
            logger.info("Computing global SHAP values...")

            # Calculate SHAP values for test set
            shap_values = self.explainer.shap_values(X_test)

            # Handle different SHAP output formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification: shap_values[1] for positive class
                shap_values_positive = shap_values[1]
            else:
                shap_values_positive = shap_values

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                shap_values_positive, self.feature_names
            )

            # Calculate SHAP summary statistics
            shap_summary = self._calculate_shap_summary(shap_values_positive, X_test)

            global_results = {
                'shap_values': shap_values_positive,
                'feature_importance': feature_importance,
                'shap_summary': shap_summary,
                'expected_value': self.explainer.expected_value
            }

            logger.info("Global SHAP explanation completed")
            return global_results

        except Exception as e:
            logger.error(f"Error in global SHAP explanation: {e}")
            raise

    def explain_local(self, student_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate local SHAP explanation for a single student.

        Args:
            student_features: Single student features (1 row DataFrame)

        Returns:
            Dictionary with local SHAP analysis for the student
        """
        try:
            logger.info("Computing local SHAP values for student...")

            # Ensure single row
            if len(student_features) != 1:
                raise ValueError("student_features must contain exactly 1 row")

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(student_features)

            # Handle different SHAP output formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_student = shap_values[1][0]  # Positive class, first row
            else:
                shap_values_student = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Get feature contributions
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_values_student,
                'feature_value': student_features.iloc[0].values
            }).sort_values('shap_value', key=abs, ascending=False)

            # Calculate prediction probability
            if hasattr(self.model, 'predict_proba'):
                prediction_prob = self.model.predict_proba(student_features)[0, 1]
            else:
                prediction_prob = self.model.predict(student_features)[0]

            local_results = {
                'shap_values': shap_values_student,
                'feature_contributions': feature_contributions,
                'prediction_probability': prediction_prob,
                'expected_value': self.explainer.expected_value,
                'student_features': student_features.iloc[0].to_dict()
            }

            logger.info("Local SHAP explanation completed")
            return local_results

        except Exception as e:
            logger.error(f"Error in local SHAP explanation: {e}")
            raise

    def get_top_features(self, shap_values: np.ndarray, feature_names: List[str],
                        top_n: int = 5) -> pd.DataFrame:
        """
        Get top N most important features based on SHAP values.

        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with top features and their importance
        """
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)

            return importance_df.head(top_n)

        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            raise

    def visualize_summary_plot(self, shap_values: np.ndarray, X_test: pd.DataFrame,
                             max_display: int = 10, show: bool = False,
                             save_path: Optional[str] = None) -> None:
        """
        Create SHAP summary plot.

        Args:
            shap_values: SHAP values
            X_test: Test features
            max_display: Maximum features to display
            show: Whether to show the plot
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, X_test,
                max_display=max_display,
                show=show
            )

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"SHAP summary plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")
            raise

    def visualize_waterfall_plot(self, student_index: int, shap_values: np.ndarray,
                               X_test: pd.DataFrame, max_display: int = 10,
                               show: bool = False, save_path: Optional[str] = None) -> None:
        """
        Create SHAP waterfall plot for a single prediction.

        Args:
            student_index: Index of the student in the test set
            shap_values: SHAP values for the test set
            X_test: Test features
            max_display: Maximum features to display
            show: Whether to show the plot
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                self.explainer.expected_value,
                shap_values[student_index],
                X_test.iloc[student_index],
                max_display=max_display,
                show=show
            )

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"SHAP waterfall plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            raise

    def visualize_bar_plot(self, shap_values: np.ndarray, X_test: pd.DataFrame,
                          max_display: int = 10, show: bool = False,
                          save_path: Optional[str] = None) -> None:
        """
        Create SHAP bar plot showing mean absolute SHAP values.

        Args:
            shap_values: SHAP values
            X_test: Test features
            max_display: Maximum features to display
            show: Whether to show the plot
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(
                self.explainer.expected_value,
                shap_values,
                feature_names=X_test.columns,
                max_display=max_display,
                show=show
            )

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"SHAP bar plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Error creating bar plot: {e}")
            raise

    def _calculate_feature_importance(self, shap_values: np.ndarray,
                                    feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values.

        Args:
            shap_values: SHAP values array
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap,
            'std_shap': np.abs(shap_values).std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        return importance_df

    def _calculate_shap_summary(self, shap_values: np.ndarray,
                              X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for SHAP values.

        Args:
            shap_values: SHAP values array
            X_test: Test features

        Returns:
            Dictionary with SHAP summary statistics
        """
        summary = {
            'mean_shap': shap_values.mean(axis=0).tolist(),
            'std_shap': shap_values.std(axis=0).tolist(),
            'min_shap': shap_values.min(axis=0).tolist(),
            'max_shap': shap_values.max(axis=0).tolist(),
            'shap_range': (shap_values.max() - shap_values.min()),
            'feature_correlation': {}
        }

        # Calculate correlation between SHAP values and feature values
        for i, feature in enumerate(X_test.columns):
            correlation = np.corrcoef(shap_values[:, i], X_test[feature].values)[0, 1]
            summary['feature_correlation'][feature] = correlation

        return summary

    def save_explanation(self, explanation: Dict[str, Any], filepath: str) -> None:
        """
        Save explanation results to file.

        Args:
            explanation: Explanation dictionary
            filepath: Path to save the explanation
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            serializable_explanation = {}
            for key, value in explanation.items():
                if isinstance(value, np.ndarray):
                    serializable_explanation[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    serializable_explanation[key] = value.to_dict()
                else:
                    serializable_explanation[key] = value

            import json
            with open(filepath, 'w') as f:
                json.dump(serializable_explanation, f, indent=2)

            logger.info(f"Explanation saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving explanation: {e}")
            raise
