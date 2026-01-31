"""
LIME Explainer for Student Suicide Risk Prediction

This module provides LIME-based explainability as a complementary approach
to SHAP for local explanations of model predictions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class LIMEExplainer:
    """
    LIME-based explainer for suicide risk prediction models.

    Provides local explanations using LIME as an alternative to SHAP.
    """

    def __init__(self, X_train: pd.DataFrame, model: Any, mode: str = 'classification',
                 feature_names: Optional[List[str]] = None, class_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.

        Args:
            X_train: Training features for LIME explainer
            model: Trained model to explain
            mode: Explanation mode ('classification' or 'regression')
            feature_names: List of feature names (optional)
            class_names: List of class names (optional)
        """
        self.X_train = X_train
        self.model = model
        self.mode = mode
        self.feature_names = feature_names or X_train.columns.tolist()
        self.class_names = class_names or ['Low Risk', 'High Risk']

        # Initialize LIME explainer
        try:
            self.explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )

            logger.info("LIME explainer initialized")

        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {e}")
            raise

    def explain_instance(self, instance: pd.DataFrame, num_features: int = 10,
                        num_samples: int = 5000) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.

        Args:
            instance: Single instance to explain (1 row DataFrame)
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME

        Returns:
            Dictionary with LIME explanation results
        """
        try:
            logger.info("Generating LIME explanation for instance...")

            # Ensure single row
            if len(instance) != 1:
                raise ValueError("instance must contain exactly 1 row")

            # Convert to numpy array
            instance_array = instance.values[0]

            # Generate explanation
            explanation = self.explainer.explain_instance(
                instance_array,
                self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )

            # Extract explanation components
            feature_importance = self._parse_lime_explanation(explanation)

            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                prediction_prob = self.model.predict_proba(instance)[0, 1]
            else:
                prediction_prob = self.model.predict(instance)[0]

            lime_results = {
                'feature_importance': feature_importance,
                'prediction_probability': prediction_prob,
                'instance_features': instance.iloc[0].to_dict(),
                'explanation_object': explanation,
                'lime_score': explanation.score
            }

            logger.info("LIME explanation generated")
            return lime_results

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            raise

    def explain_multiple_instances(self, instances: pd.DataFrame, num_features: int = 10,
                                 num_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for multiple instances.

        Args:
            instances: DataFrame with multiple instances to explain
            num_features: Number of features per explanation
            num_samples: Number of samples per explanation

        Returns:
            List of explanation dictionaries
        """
        try:
            logger.info(f"Generating LIME explanations for {len(instances)} instances...")

            explanations = []
            for i in range(len(instances)):
                instance_df = instances.iloc[[i]]
                explanation = self.explain_instance(
                    instance_df, num_features=num_features, num_samples=num_samples
                )
                explanations.append(explanation)

            logger.info("Multiple LIME explanations completed")
            return explanations

        except Exception as e:
            logger.error(f"Error generating multiple LIME explanations: {e}")
            raise

    def _parse_lime_explanation(self, explanation) -> pd.DataFrame:
        """
        Parse LIME explanation object into structured format.

        Args:
            explanation: LIME explanation object

        Returns:
            DataFrame with feature importance
        """
        try:
            # Get feature importance as list of tuples
            feature_importance_list = explanation.as_list()

            # Convert to DataFrame
            features = []
            weights = []

            for feature, weight in feature_importance_list:
                features.append(feature)
                weights.append(weight)

            importance_df = pd.DataFrame({
                'feature': features,
                'weight': weights,
                'abs_weight': np.abs(weights)
            }).sort_values('abs_weight', ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Error parsing LIME explanation: {e}")
            raise

    def visualize_explanation(self, explanation: Dict[str, Any], show: bool = False,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize LIME explanation.

        Args:
            explanation: LIME explanation dictionary
            show: Whether to show the plot
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 6))

            # Get explanation object
            exp_obj = explanation['explanation_object']

            # Create visualization
            exp_obj.as_pyplot_figure()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"LIME explanation plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            logger.error(f"Error visualizing LIME explanation: {e}")
            raise

    def compare_with_shap(self, lime_explanation: Dict[str, Any],
                         shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare LIME and SHAP explanations for the same instance.

        Args:
            lime_explanation: LIME explanation dictionary
            shap_explanation: SHAP explanation dictionary

        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Extract top features from both explanations
            lime_features = set(lime_explanation['feature_importance']['feature'].head(5))
            shap_features = set(shap_explanation['feature_contributions']['feature'].head(5))

            # Calculate agreement metrics
            intersection = lime_features.intersection(shap_features)
            union = lime_features.union(shap_features)

            jaccard_similarity = len(intersection) / len(union) if union else 0

            comparison = {
                'lime_top_features': list(lime_features),
                'shap_top_features': list(shap_features),
                'common_features': list(intersection),
                'jaccard_similarity': jaccard_similarity,
                'agreement_score': len(intersection) / 5  # Fraction of top 5 that agree
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing LIME and SHAP: {e}")
            raise

    def get_explanation_stability(self, instance: pd.DataFrame, n_runs: int = 10,
                                num_features: int = 10) -> Dict[str, Any]:
        """
        Assess stability of LIME explanations across multiple runs.

        Args:
            instance: Single instance to explain
            n_runs: Number of explanation runs
            num_features: Number of features per explanation

        Returns:
            Dictionary with stability metrics
        """
        try:
            logger.info(f"Assessing LIME explanation stability over {n_runs} runs...")

            all_explanations = []
            for i in range(n_runs):
                explanation = self.explain_instance(instance, num_features=num_features)
                all_explanations.append(explanation)

            # Extract feature rankings
            feature_rankings = []
            for exp in all_explanations:
                ranking = exp['feature_importance']['feature'].tolist()
                feature_rankings.append(ranking)

            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(feature_rankings)

            stability_results = {
                'n_runs': n_runs,
                'stability_metrics': stability_metrics,
                'all_rankings': feature_rankings
            }

            logger.info("LIME stability assessment completed")
            return stability_results

        except Exception as e:
            logger.error(f"Error assessing LIME stability: {e}")
            raise

    def _calculate_stability_metrics(self, rankings: List[List[str]]) -> Dict[str, Any]:
        """
        Calculate stability metrics for feature rankings.

        Args:
            rankings: List of feature ranking lists

        Returns:
            Dictionary with stability metrics
        """
        try:
            n_runs = len(rankings)
            n_features = len(rankings[0])

            # Calculate pairwise agreement
            agreements = []
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    rank1 = rankings[i][:5]  # Top 5 features
                    rank2 = rankings[j][:5]
                    agreement = len(set(rank1).intersection(set(rank2))) / 5
                    agreements.append(agreement)

            # Calculate rank correlation (simplified)
            rank_stability = np.mean(agreements)

            stability = {
                'mean_top5_agreement': rank_stability,
                'std_top5_agreement': np.std(agreements),
                'min_top5_agreement': np.min(agreements),
                'max_top5_agreement': np.max(agreements)
            }

            return stability

        except Exception as e:
            logger.error(f"Error calculating stability metrics: {e}")
            raise

    def save_explanation(self, explanation: Dict[str, Any], filepath: str) -> None:
        """
        Save LIME explanation results to file.

        Args:
            explanation: LIME explanation dictionary
            filepath: Path to save the explanation
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Remove non-serializable objects
            serializable_explanation = explanation.copy()
            if 'explanation_object' in serializable_explanation:
                del serializable_explanation['explanation_object']

            # Convert DataFrames to dict
            if 'feature_importance' in serializable_explanation:
                serializable_explanation['feature_importance'] = \
                    serializable_explanation['feature_importance'].to_dict()

            import json
            with open(filepath, 'w') as f:
                json.dump(serializable_explanation, f, indent=2)

            logger.info(f"LIME explanation saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving LIME explanation: {e}")
            raise
