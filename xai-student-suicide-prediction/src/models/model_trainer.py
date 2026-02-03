"""
Model Training Pipeline for Student Suicide Risk Prediction

This module provides a complete training pipeline that:
- Loads and preprocesses multi-modal data
- Applies SMOTE for class imbalance
- Trains Random Forest and XGBoost models
- Performs cross-validation and hyperparameter tuning
- Saves trained models for deployment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import joblib

from data.data_loader import DataLoader
from data.data_fusion import merge_multimodal_data
from data.preprocessor import DataPreprocessor
from .random_forest import RFModel
from .xgboost_model import XGBoostModel

# Set up logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Complete training pipeline for suicide risk prediction models.

    Orchestrates the entire training process from data loading to model saving.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or self._get_default_config()
        self.data_loader = DataLoader(self.config['data']['data_dir'])
        self.preprocessor = DataPreprocessor(random_state=self.config['data']['random_state'])

        # Initialize models
        self.rf_model = RFModel(
            n_estimators=self.config['model']['rf']['n_estimators'],
            max_depth=self.config['model']['rf']['max_depth'],
            min_samples_split=self.config['model']['rf']['min_samples_split'],
            random_state=self.config['data']['random_state']
        )

        self.xgb_model = XGBoostModel(
            n_estimators=self.config['model']['xgboost']['n_estimators'],
            max_depth=self.config['model']['xgboost']['max_depth'],
            learning_rate=self.config['model']['xgboost']['learning_rate'],
            scale_pos_weight=self.config['model']['xgboost']['scale_pos_weight'],
            random_state=self.config['data']['random_state']
        )

        self.trained_models = {}
        logger.info("ModelTrainer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for training."""
        return {
            'data': {
                'data_dir': 'data/synthetic',
                'train_split': 0.70,
                'val_split': 0.15,
                'test_split': 0.15,
                'random_state': 42
            },
            'model': {
                'rf': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 19  # For ~5% minority class
                }
            },
            'preprocessing': {
                'apply_smote': True,
                'smote_sampling_strategy': 0.3
            },
            'training': {
                'cv_folds': 5,
                'hyperparameter_tuning': False
            },
            'output': {
                'model_dir': 'models',
                'results_dir': 'results'
            }
        }

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and preprocess the complete dataset.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Loading and preprocessing data...")

            # Load all data modalities
            behavioral_df, academic_df, engagement_df, risk_df = self.data_loader.load_all_data()

            # Merge multimodal data
            fused_df = merge_multimodal_data(
                behavioral_df, academic_df, engagement_df, risk_df,
                imputation_strategy='mean'
            )

            logger.info(f"Fused dataset shape: {fused_df.shape}")

            # Split features and target
            feature_cols = [col for col in fused_df.columns if col not in ['student_id', 'risk_label']]
            X = fused_df[feature_cols]
            y = fused_df['risk_label']

            # Train/validation/test split
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.stratified_split(
                X, y,
                train_size=self.config['data']['train_split'],
                val_size=self.config['data']['val_split'],
                test_size=self.config['data']['test_split']
            )

            logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
            logger.info(f"Train labels - High risk: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")

            # Apply SMOTE to training set only
            if self.config['preprocessing']['apply_smote']:
                logger.info("Applying SMOTE to training set...")
                X_train_smote, y_train_smote = self.preprocessor.apply_smote(
                    X_train, y_train,
                    sampling_strategy=self.config['preprocessing']['smote_sampling_strategy']
                )
                logger.info(f"After SMOTE - Train set: {X_train_smote.shape}")
                logger.info(f"After SMOTE - High risk: {y_train_smote.sum()}/{len(y_train_smote)} ({y_train_smote.mean():.1%})")

                X_train, y_train = X_train_smote, y_train_smote

            # Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.preprocessor.fit_transform(X_train)
            X_val_scaled = self.preprocessor.transform(X_val)
            X_test_scaled = self.preprocessor.transform(X_test)

            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error in data loading and preprocessing: {e}")
            raise

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train both Random Forest and XGBoost models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Starting model training...")

            training_results = {}

            # Train Random Forest
            logger.info("Training Random Forest model...")
            self.rf_model.train(X_train, y_train)
            training_results['rf'] = {'status': 'trained'}

            # Train XGBoost
            logger.info("Training XGBoost model...")
            self.xgb_model.train(X_train, y_train, X_val, y_val)
            training_results['xgb'] = {'status': 'trained'}

            # Store trained models
            self.trained_models = {
                'rf': self.rf_model,
                'xgb': self.xgb_model,
                'preprocessor': self.preprocessor
            }

            logger.info("Model training completed")
            return training_results

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate both trained models on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info("Evaluating models on test set...")

            evaluation_results = {}

            # Evaluate Random Forest
            if 'rf' in self.trained_models:
                logger.info("Evaluating Random Forest...")
                rf_eval = self.trained_models['rf'].evaluate(X_test, y_test, threshold)
                evaluation_results['rf'] = rf_eval

            # Evaluate XGBoost
            if 'xgb' in self.trained_models:
                logger.info("Evaluating XGBoost...")
                xgb_eval = self.trained_models['xgb'].evaluate(X_test, y_test, threshold)
                evaluation_results['xgb'] = xgb_eval

            # Compare models
            if 'rf' in evaluation_results and 'xgb' in evaluation_results:
                rf_f1 = evaluation_results['rf']['classification_report']['weighted avg']['f1-score']
                xgb_f1 = evaluation_results['xgb']['classification_report']['weighted avg']['f1-score']

                logger.info(f"Random Forest F1: {rf_f1:.4f}")
                logger.info(f"XGBoost F1: {xgb_f1:.4f}")

                if xgb_f1 > rf_f1:
                    logger.info("XGBoost performed better")
                    evaluation_results['best_model'] = 'xgb'
                else:
                    logger.info("Random Forest performed better")
                    evaluation_results['best_model'] = 'rf'

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise

    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation on both models.

        Args:
            X: Feature data
            y: Target labels

        Returns:
            Cross-validation results
        """
        try:
            logger.info("Performing cross-validation...")

            cv_results = {}

            # Cross-validate Random Forest
            logger.info("Cross-validating Random Forest...")
            rf_cv = self.rf_model.cross_validate(X, y, cv=self.config['training']['cv_folds'])
            cv_results['rf'] = rf_cv

            # Cross-validate XGBoost
            logger.info("Cross-validating XGBoost...")
            xgb_cv = self.xgb_model.cross_validate(X, y, cv=self.config['training']['cv_folds'])
            cv_results['xgb'] = xgb_cv

            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning on both models.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Hyperparameter tuning results
        """
        try:
            logger.info("Starting hyperparameter tuning...")

            tuning_results = {}

            # Tune Random Forest
            logger.info("Tuning Random Forest hyperparameters...")
            rf_tuning = self.rf_model.hyperparameter_tuning(X_train, y_train)
            tuning_results['rf'] = rf_tuning

            # Tune XGBoost
            logger.info("Tuning XGBoost hyperparameters...")
            xgb_tuning = self.xgb_model.hyperparameter_tuning(X_train, y_train)
            tuning_results['xgb'] = xgb_tuning

            return tuning_results

        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise

    def save_models(self, model_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save trained models to disk.

        Args:
            model_dir: Directory to save models (optional)

        Returns:
            Dictionary with saved model paths
        """
        try:
            model_dir = model_dir or self.config['output']['model_dir']
            Path(model_dir).mkdir(parents=True, exist_ok=True)

            saved_paths = {}

            # Save Random Forest
            if 'rf' in self.trained_models:
                rf_path = Path(model_dir) / 'random_forest_model.pkl'
                self.trained_models['rf'].save_model(str(rf_path))
                saved_paths['rf'] = str(rf_path)

            # Save XGBoost
            if 'xgb' in self.trained_models:
                xgb_path = Path(model_dir) / 'xgboost_model.pkl'
                self.trained_models['xgb'].save_model(str(xgb_path))
                saved_paths['xgb'] = str(xgb_path)

            # Save preprocessor
            if 'preprocessor' in self.trained_models:
                prep_path = Path(model_dir) / 'preprocessor.pkl'
                joblib.dump(self.trained_models['preprocessor'], prep_path)
                saved_paths['preprocessor'] = str(prep_path)

            logger.info(f"Models saved to {model_dir}")
            return saved_paths

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data loading to model saving.

        Returns:
            Dictionary with complete training results
        """
        try:
            logger.info("Starting complete training pipeline...")

            # Load and preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()

            # Train models
            training_results = self.train_models(X_train, y_train, X_val, y_val)

            # Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test)

            # Perform cross-validation
            cv_results = self.perform_cross_validation(
                pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
            )

            # Hyperparameter tuning (optional)
            tuning_results = None
            if self.config['training']['hyperparameter_tuning']:
                tuning_results = self.hyperparameter_tuning(X_train, y_train)

            # Save models
            saved_paths = self.save_models()

            # Compile complete results
            complete_results = {
                'training': training_results,
                'evaluation': evaluation_results,
                'cross_validation': cv_results,
                'hyperparameter_tuning': tuning_results,
                'saved_models': saved_paths,
                'data_shapes': {
                    'train': X_train.shape,
                    'validation': X_val.shape,
                    'test': X_test.shape
                },
                'class_distribution': {
                    'train': {'high_risk': y_train.sum(), 'total': len(y_train)},
                    'validation': {'high_risk': y_val.sum(), 'total': len(y_val)},
                    'test': {'high_risk': y_test.sum(), 'total': len(y_test)}
                }
            }

            logger.info("Complete training pipeline finished successfully")
            return complete_results

        except Exception as e:
            logger.error(f"Error in complete training pipeline: {e}")
            raise

def train_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete model training.

    Args:
        config: Training configuration (optional)

    Returns:
        Complete training results
    """
    trainer = ModelTrainer(config)
    return trainer.run_complete_training_pipeline()
