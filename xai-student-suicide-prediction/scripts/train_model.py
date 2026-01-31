#!/usr/bin/env python3
"""
Model Training Script for Student Suicide Risk Prediction

This script runs the complete training pipeline:
1. Load and preprocess data
2. Train Random Forest and XGBoost models
3. Evaluate models
4. Save trained models

Usage:
    python scripts/train_model.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.model_trainer import ModelTrainer
from evaluation.metrics import generate_evaluation_report
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        logger.info("Starting model training pipeline...")

        # Initialize trainer with default config
        trainer = ModelTrainer()

        # Run complete training pipeline
        results = trainer.run_complete_training_pipeline()

        # Log key results
        logger.info("Training completed successfully!")
        logger.info(f"Data shapes: {results['data_shapes']}")
        logger.info(f"Models saved: {list(results['saved_models'].keys())}")

        # Log evaluation results
        rf_eval = results['evaluation']['rf']
        xgb_eval = results['evaluation']['xgb']

        logger.info("Random Forest Results:")
        logger.info(f"  AUC-ROC: {rf_eval['auc_roc']:.4f}")
        logger.info(f"  F1-Score: {rf_eval['f1']:.4f}")
        logger.info(f"  Precision: {rf_eval['precision']:.4f}")
        logger.info(f"  Recall: {rf_eval['recall']:.4f}")

        logger.info("XGBoost Results:")
        logger.info(f"  AUC-ROC: {xgb_eval['auc_roc']:.4f}")
        logger.info(f"  F1-Score: {xgb_eval['f1']:.4f}")
        logger.info(f"  Precision: {xgb_eval['precision']:.4f}")
        logger.info(f"  Recall: {xgb_eval['recall']:.4f}")

        # Determine best model
        best_model = results['evaluation']['best_model']
        logger.info(f"Best performing model: {best_model}")

        # Save detailed results
        results_file = Path('results/training_results.json')
        results_file.parent.mkdir(exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Detailed results saved to {results_file}")

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total students processed: {results['data_shapes']['train'][0] + results['data_shapes']['validation'][0] + results['data_shapes']['test'][0]}")
        print(f"Training set size: {results['data_shapes']['train'][0]}")
        print(f"Features used: {results['data_shapes']['train'][1]}")
        print()
        print("MODEL PERFORMANCE:")
        print(f"Random Forest AUC-ROC: {rf_eval['auc_roc']:.4f}")
        print(f"XGBoost AUC-ROC: {xgb_eval['auc_roc']:.4f}")
        print(f"Best model: {best_model.upper()}")
        print()
        print("TARGET METRICS CHECK:")
        print(f"AUC-ROC ≥ 0.85: {'✓' if max(rf_eval['auc_roc'], xgb_eval['auc_roc']) >= 0.85 else '✗'}")
        print(f"Recall ≥ 0.75: {'✓' if max(rf_eval['recall'], xgb_eval['recall']) >= 0.75 else '✗'}")
        print(f"Precision ≥ 0.60: {'✓' if max(rf_eval['precision'], xgb_eval['precision']) >= 0.60 else '✗'}")
        print(f"F1-Score ≥ 0.70: {'✓' if max(rf_eval['f1'], xgb_eval['f1']) >= 0.70 else '✗'}")
        print("="*60)

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
