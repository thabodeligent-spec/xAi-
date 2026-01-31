"""
XGBoost Model for Student Suicide Risk Prediction

This module implements an XGBoost classifier optimized for imbalanced data
with scale_pos_weight for class imbalance and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    XGBoost model for suicide risk prediction.

    Optimized for imbalanced datasets using scale_pos_weight
    and early stopping for better performance.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 scale_pos_weight: Optional[float] = None,
                 random_state: int = 42):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of each tree
            learning_rate: Learning rate (eta)
            scale_pos_weight: Weight for positive class (for imbalanced data)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

        self.is_trained = False
        self.feature_names = None
        logger.info(f"XGBoost model initialized with {n_estimators} estimators")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              early_stopping_rounds: int = 10) -> None:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            early_stopping_rounds: Number of rounds for early stopping
        """
        try:
            logger.info("Training XGBoost model...")

            # Store feature names for later use
            self.feature_names = X_train.columns.tolist()

            # Prepare evaluation set for early stopping
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
                logger.info("Using early stopping with validation set")
            else:
                eval_set = [(X_train, y_train)]

            # Train the model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

            self.is_trained = True
            logger.info("XGBoost training completed")

            # Log best iteration if early stopping was used
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"Best iteration: {self.model.best_iteration}")

            # Log feature importances
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                logger.info("Top 5 features by importance:")
                for _, row in feature_importance.head(5).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features

        Returns:
            Predicted probabilities for positive class

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Ensure feature order matches training
            if self.feature_names:
                X_test = X_test[self.feature_names]

            probabilities = self.model.predict_proba(X_test)[:, 1]
            return probabilities

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_classes(self, X_test: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make class predictions with custom threshold.

        Args:
            X_test: Test features
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        probabilities = self.predict(X_test)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            probabilities = self.predict(X_test)
            predictions = (probabilities >= threshold).astype(int)

            # Calculate metrics
            report = classification_report(y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predictions)

            evaluation_results = {
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
                'model_params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'scale_pos_weight': self.scale_pos_weight
                }
            }

            logger.info(f"Model evaluation completed. F1-score: {report['weighted avg']['f1-score']:.4f}")
            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.

        Args:
            X: Feature data
            y: Target labels
            cv: Number of cross-validation folds

        Returns:
            Cross-validation results
        """
        try:
            logger.info(f"Performing {cv}-fold cross-validation...")

            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )

            cv_results = {
                'cv_accuracy': cv_scores[0].tolist(),
                'cv_precision': cv_scores[1].tolist(),
                'cv_recall': cv_scores[2].tolist(),
                'cv_f1': cv_scores[3].tolist(),
                'cv_roc_auc': cv_scores[4].tolist(),
                'mean_accuracy': cv_scores[0].mean(),
                'mean_precision': cv_scores[1].mean(),
                'mean_recall': cv_scores[2].mean(),
                'mean_f1': cv_scores[3].mean(),
                'mean_roc_auc': cv_scores[4].mean(),
                'std_accuracy': cv_scores[0].std(),
                'std_precision': cv_scores[1].std(),
                'std_recall': cv_scores[2].std(),
                'std_f1': cv_scores[3].std(),
                'std_roc_auc': cv_scores[4].std()
            }

            logger.info(f"Mean F1: {cv_results['mean_f1']:.4f}")
            logger.info(f"Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f}")
            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning (optional)

        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

        try:
            logger.info("Starting hyperparameter tuning...")

            grid_search = GridSearchCV(
                XGBClassifier(
                    scale_pos_weight=self.scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.max_depth = grid_search.best_params_['max_depth']
            self.learning_rate = grid_search.best_params_['learning_rate']

            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
            return tuning_results

        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save model and metadata
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'scale_pos_weight': self.scale_pos_weight,
                    'random_state': self.random_state
                }
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = True

            # Update parameters
            params = model_data['params']
            self.n_estimators = params['n_estimators']
            self.max_depth = params['max_depth']
            self.learning_rate = params['learning_rate']
            self.scale_pos_weight = params['scale_pos_weight']
            self.random_state = params['random_state']

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance rankings.

        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def get_booster(self):
        """
        Get the underlying XGBoost booster for advanced operations.

        Returns:
            XGBoost booster object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        return self.model.get_booster()
