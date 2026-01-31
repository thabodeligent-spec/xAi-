"""
Fairness Audit System for Student Suicide Risk Prediction

This module provides fairness assessment tools to ensure equitable model performance
across different demographic groups and prevent bias in suicide risk predictions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class FairnessAuditor:
    """
    Comprehensive fairness audit system for suicide risk prediction models.

    Assesses model fairness across protected attributes and demographic groups.
    """

    def __init__(self, protected_attributes: Optional[List[str]] = None,
                 fairness_threshold: float = 0.05):
        """
        Initialize fairness auditor.

        Args:
            protected_attributes: List of protected attribute column names
            fairness_threshold: Threshold for flagging fairness violations
        """
        self.protected_attributes = protected_attributes or ['gender', 'ethnicity', 'age_group', 'socioeconomic_status']
        self.fairness_threshold = fairness_threshold
        logger.info(f"FairnessAuditor initialized with threshold: {fairness_threshold}")

    def audit_model_fairness(self, X_test: pd.DataFrame, y_true: np.ndarray,
                           y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform comprehensive fairness audit on model predictions.

        Args:
            X_test: Test features (including protected attributes)
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary with fairness audit results
        """
        try:
            logger.info("Starting comprehensive fairness audit...")

            # Convert predictions
            y_pred = (y_pred_proba >= threshold).astype(int)

            audit_results = {
                'overall_metrics': self._calculate_overall_metrics(y_true, y_pred, y_pred_proba),
                'protected_attribute_analysis': {},
                'fairness_violations': [],
                'recommendations': []
            }

            # Analyze each protected attribute
            for attr in self.protected_attributes:
                if attr in X_test.columns:
                    attr_results = self._analyze_protected_attribute(
                        X_test[attr], y_true, y_pred, y_pred_proba, attr
                    )
                    audit_results['protected_attribute_analysis'][attr] = attr_results

                    # Check for fairness violations
                    violations = self._check_fairness_violations(attr_results, attr)
                    audit_results['fairness_violations'].extend(violations)
                else:
                    logger.warning(f"Protected attribute '{attr}' not found in test data")

            # Generate recommendations
            audit_results['recommendations'] = self._generate_fairness_recommendations(
                audit_results['fairness_violations']
            )

            # Overall fairness assessment
            audit_results['overall_fairness'] = self._assess_overall_fairness(audit_results)

            logger.info("Fairness audit completed")
            return audit_results

        except Exception as e:
            logger.error(f"Error in fairness audit: {e}")
            raise

    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate overall model performance metrics."""
        from .metrics import evaluate_model
        return evaluate_model(y_true, y_pred_proba, threshold=0.5)

    def _analyze_protected_attribute(self, protected_attr: pd.Series, y_true: np.ndarray,
                                   y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                   attr_name: str) -> Dict[str, Any]:
        """
        Analyze model performance across different groups of a protected attribute.

        Args:
            protected_attr: Values of the protected attribute
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            attr_name: Name of the protected attribute

        Returns:
            Dictionary with group-wise analysis
        """
        try:
            # Get unique groups
            groups = protected_attr.dropna().unique()

            group_analysis = {
                'attribute_name': attr_name,
                'groups': {},
                'disparity_metrics': {}
            }

            group_metrics = []

            for group in groups:
                # Get indices for this group
                group_mask = protected_attr == group
                group_size = group_mask.sum()

                if group_size < 10:  # Skip very small groups
                    continue

                # Calculate metrics for this group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                group_y_pred_proba = y_pred_proba[group_mask]

                # Basic metrics
                from .metrics import evaluate_model
                metrics = evaluate_model(group_y_true, group_y_pred_proba, threshold=0.5)

                # Group representation
                total_samples = len(y_true)
                group_prevalence = group_y_true.mean()
                overall_prevalence = y_true.mean()

                group_info = {
                    'group_size': int(group_size),
                    'group_percentage': group_size / total_samples,
                    'true_prevalence': group_prevalence,
                    'predicted_prevalence': group_y_pred.mean(),
                    'metrics': metrics
                }

                group_analysis['groups'][str(group)] = group_info
                group_metrics.append(metrics)

            # Calculate disparity metrics
            if len(group_metrics) > 1:
                group_analysis['disparity_metrics'] = self._calculate_disparity_metrics(
                    group_metrics, groups
                )

            return group_analysis

        except Exception as e:
            logger.error(f"Error analyzing protected attribute {attr_name}: {e}")
            raise

    def _calculate_disparity_metrics(self, group_metrics: List[Dict], groups: np.ndarray) -> Dict[str, Any]:
        """Calculate fairness disparity metrics across groups."""
        try:
            # Extract key metrics
            auc_scores = [m['auc_roc'] for m in group_metrics]
            precision_scores = [m['precision'] for m in group_metrics]
            recall_scores = [m['recall'] for m in group_metrics]
            f1_scores = [m['f1'] for m in group_metrics]

            disparity_metrics = {
                'auc_range': max(auc_scores) - min(auc_scores),
                'auc_std': np.std(auc_scores),
                'precision_range': max(precision_scores) - min(precision_scores),
                'precision_std': np.std(precision_scores),
                'recall_range': max(recall_scores) - min(recall_scores),
                'recall_std': np.std(recall_scores),
                'f1_range': max(f1_scores) - min(f1_scores),
                'f1_std': np.std(f1_scores),
                'groups_analyzed': len(group_metrics)
            }

            return disparity_metrics

        except Exception as e:
            logger.error(f"Error calculating disparity metrics: {e}")
            raise

    def _check_fairness_violations(self, attr_results: Dict, attr_name: str) -> List[Dict]:
        """Check for fairness violations in protected attribute analysis."""
        try:
            violations = []

            disparity = attr_results.get('disparity_metrics', {})

            # Check AUC disparity
            if disparity.get('auc_range', 0) > self.fairness_threshold:
                violations.append({
                    'type': 'auc_disparity',
                    'attribute': attr_name,
                    'severity': 'high' if disparity['auc_range'] > 0.1 else 'medium',
                    'value': disparity['auc_range'],
                    'threshold': self.fairness_threshold,
                    'description': f"AUC disparity across {attr_name} groups exceeds threshold"
                })

            # Check F1 disparity
            if disparity.get('f1_range', 0) > self.fairness_threshold:
                violations.append({
                    'type': 'f1_disparity',
                    'attribute': attr_name,
                    'severity': 'high' if disparity['f1_range'] > 0.1 else 'medium',
                    'value': disparity['f1_range'],
                    'threshold': self.fairness_threshold,
                    'description': f"F1-score disparity across {attr_name} groups exceeds threshold"
                })

            # Check for underrepresented groups
            groups = attr_results.get('groups', {})
            for group_name, group_info in groups.items():
                group_pct = group_info.get('group_percentage', 0)
                if group_pct < 0.05:  # Less than 5% representation
                    violations.append({
                        'type': 'underrepresentation',
                        'attribute': attr_name,
                        'group': group_name,
                        'severity': 'medium',
                        'value': group_pct,
                        'threshold': 0.05,
                        'description': f"Group {group_name} is underrepresented (<5% of population)"
                    })

            return violations

        except Exception as e:
            logger.error(f"Error checking fairness violations: {e}")
            raise

    def _generate_fairness_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on fairness violations."""
        try:
            recommendations = []

            violation_types = set(v['type'] for v in violations)

            if 'auc_disparity' in violation_types or 'f1_disparity' in violation_types:
                recommendations.append(
                    "Consider reweighting training data or using fairness-aware algorithms to reduce performance disparities across demographic groups"
                )

            if 'underrepresentation' in violation_types:
                recommendations.append(
                    "Collect more data from underrepresented groups to improve model generalizability and fairness"
                )

            # Severity-based recommendations
            high_severity = any(v['severity'] == 'high' for v in violations)
            if high_severity:
                recommendations.append(
                    "High fairness violations detected. Consider model retraining with fairness constraints or alternative modeling approaches"
                )

            if not recommendations:
                recommendations.append(
                    "No significant fairness violations detected. Continue monitoring model performance across demographic groups"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating fairness recommendations: {e}")
            raise

    def _assess_overall_fairness(self, audit_results: Dict) -> Dict[str, Any]:
        """Assess overall fairness of the model."""
        try:
            violations = audit_results.get('fairness_violations', [])
            high_violations = sum(1 for v in violations if v['severity'] == 'high')
            medium_violations = sum(1 for v in violations if v['severity'] == 'medium')

            # Overall fairness score (0-1, higher is better)
            fairness_score = max(0, 1 - (high_violations * 0.3 + medium_violations * 0.1))

            overall_assessment = {
                'fairness_score': fairness_score,
                'total_violations': len(violations),
                'high_severity_violations': high_violations,
                'medium_severity_violations': medium_violations,
                'fairness_rating': self._get_fairness_rating(fairness_score),
                'requires_attention': high_violations > 0 or medium_violations > 2
            }

            return overall_assessment

        except Exception as e:
            logger.error(f"Error assessing overall fairness: {e}")
            raise

    def _get_fairness_rating(self, score: float) -> str:
        """Convert fairness score to rating."""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Concerning'
        else:
            return 'Poor'

    def plot_fairness_analysis(self, audit_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Create visualizations for fairness analysis.

        Args:
            audit_results: Results from fairness audit
            save_path: Path to save the plot (optional)
        """
        try:
            protected_analysis = audit_results.get('protected_attribute_analysis', {})

            if not protected_analysis:
                logger.warning("No protected attribute analysis available for plotting")
                return

            # Create subplots for each protected attribute
            n_attrs = len(protected_analysis)
            fig, axes = plt.subplots(n_attrs, 2, figsize=(15, 5 * n_attrs))
            if n_attrs == 1:
                axes = [axes]

            for i, (attr_name, attr_results) in enumerate(protected_analysis.items()):
                groups = attr_results.get('groups', {})

                if not groups:
                    continue

                # Extract data for plotting
                group_names = []
                auc_scores = []
                f1_scores = []
                group_sizes = []

                for group_name, group_info in groups.items():
                    group_names.append(str(group_name))
                    auc_scores.append(group_info['metrics']['auc_roc'])
                    f1_scores.append(group_info['metrics']['f1'])
                    group_sizes.append(group_info['group_size'])

                # AUC comparison
                ax1 = axes[i][0] if n_attrs > 1 else axes[0]
                bars = ax1.bar(group_names, auc_scores, color='skyblue', alpha=0.7)
                ax1.axhline(y=audit_results['overall_metrics']['auc_roc'],
                           color='red', linestyle='--', label='Overall AUC')
                ax1.set_title(f'{attr_name} - AUC by Group')
                ax1.set_ylabel('AUC-ROC')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar, score in zip(bars, auc_scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            '.3f', ha='center', va='bottom')

                # F1 comparison
                ax2 = axes[i][1] if n_attrs > 1 else axes[1]
                bars = ax2.bar(group_names, f1_scores, color='lightgreen', alpha=0.7)
                ax2.axhline(y=audit_results['overall_metrics']['f1'],
                           color='red', linestyle='--', label='Overall F1')
                ax2.set_title(f'{attr_name} - F1-Score by Group')
                ax2.set_ylabel('F1-Score')
                ax2.legend()
                ax2.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar, score in zip(bars, f1_scores):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            '.3f', ha='center', va='bottom')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Fairness analysis plot saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting fairness analysis: {e}")
            raise

    def generate_fairness_report(self, audit_results: Dict, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive fairness report.

        Args:
            audit_results: Results from fairness audit
            output_dir: Directory to save report and plots (optional)

        Returns:
            Dictionary with fairness report
        """
        try:
            logger.info("Generating fairness report...")

            # Generate plots if output directory provided
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                plot_path = Path(output_dir) / "fairness_analysis.png"
                self.plot_fairness_analysis(audit_results, str(plot_path))

            report = {
                'audit_timestamp': pd.Timestamp.now().isoformat(),
                'fairness_threshold': self.fairness_threshold,
                'overall_fairness': audit_results.get('overall_fairness', {}),
                'violations_summary': {
                    'total_violations': len(audit_results.get('fairness_violations', [])),
                    'high_severity': sum(1 for v in audit_results.get('fairness_violations', [])
                                       if v['severity'] == 'high'),
                    'medium_severity': sum(1 for v in audit_results.get('fairness_violations', [])
                                         if v['severity'] == 'medium')
                },
                'protected_attributes_analyzed': list(audit_results.get('protected_attribute_analysis', {}).keys()),
                'recommendations': audit_results.get('recommendations', []),
                'plots_generated': output_dir is not None
            }

            logger.info("Fairness report generated")
            return report

        except Exception as e:
            logger.error(f"Error generating fairness report: {e}")
            raise

def audit_model_fairness(X_test: pd.DataFrame, y_true: np.ndarray, y_pred_proba: np.ndarray,
                        protected_attributes: Optional[List[str]] = None,
                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convenience function for fairness audit.

    Args:
        X_test: Test features with protected attributes
        y_true: True labels
        y_pred_proba: Predicted probabilities
        protected_attributes: List of protected attribute names
        threshold: Classification threshold

    Returns:
        Fairness audit results
    """
    auditor = FairnessAuditor(protected_attributes)
    return auditor.audit_model_fairness(X_test, y_true, y_pred_proba, threshold)
