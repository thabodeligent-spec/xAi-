"""
Data Fusion Module for Student Suicide Risk Prediction

This module provides functions for fusing multi-modal data:
- Early fusion by merging on student_id
- Missing value imputation strategies
- Data validation and integrity checks

Supports early fusion approach as recommended by literature review.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.impute import SimpleImputer

# Set up logging
logger = logging.getLogger(__name__)

class DataFusion:
    """Class for fusing multi-modal student data."""

    def __init__(self, imputation_strategy: str = 'mean'):
        """
        Initialize DataFusion with imputation strategy.

        Args:
            imputation_strategy: Strategy for missing value imputation
                               ('mean', 'median', 'most_frequent', 'constant')
        """
        self.imputation_strategy = imputation_strategy
        self.imputer = None
        logger.info(f"DataFusion initialized with imputation strategy: {imputation_strategy}")

    def merge_multimodal_data(self,
                            behavioral_df: pd.DataFrame,
                            academic_df: pd.DataFrame,
                            engagement_df: pd.DataFrame,
                            risk_df: Optional[pd.DataFrame] = None,
                            how: str = 'inner') -> pd.DataFrame:
        """
        Perform early fusion of multi-modal data on student_id.

        Args:
            behavioral_df: Behavioral features DataFrame
            academic_df: Academic features DataFrame
            engagement_df: Engagement features DataFrame
            risk_df: Risk labels DataFrame (optional)
            how: Merge type ('inner', 'outer', 'left', 'right')

        Returns:
            Fused DataFrame with all features

        Raises:
            ValueError: If student_id columns are missing or merge fails
        """
        try:
            # Validate student_id presence
            dataframes = [behavioral_df, academic_df, engagement_df]
            if risk_df is not None:
                dataframes.append(risk_df)

            for i, df in enumerate(dataframes):
                if 'student_id' not in df.columns:
                    raise ValueError(f"DataFrame {i} missing 'student_id' column")

            # Start with behavioral data as base
            logger.info("Starting early fusion with behavioral data")
            fused_df = behavioral_df.copy()

            # Merge academic data
            logger.info("Merging academic data...")
            fused_df = fused_df.merge(academic_df, on='student_id', how=how)
            logger.info(f"Fused data shape after academic merge: {fused_df.shape}")

            # Merge engagement data
            logger.info("Merging engagement data...")
            fused_df = fused_df.merge(engagement_df, on='student_id', how=how)
            logger.info(f"Fused data shape after engagement merge: {fused_df.shape}")

            # Merge risk labels if provided
            if risk_df is not None:
                logger.info("Merging risk labels...")
                fused_df = fused_df.merge(risk_df, on='student_id', how=how)
                logger.info(f"Final fused data shape: {fused_df.shape}")

            # Validate merge result
            self._validate_fused_data(fused_df, risk_df is not None)

            logger.info(f"Successfully fused {len(fused_df)} student records")
            return fused_df

        except Exception as e:
            logger.error(f"Error during data fusion: {e}")
            raise

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with imputed values
        """
        try:
            # Check for missing values
            missing_summary = df.isnull().sum()
            total_missing = missing_summary.sum()

            if total_missing == 0:
                logger.info("No missing values found in dataset")
                return df

            logger.info(f"Found {total_missing} missing values across {len(missing_summary[missing_summary > 0])} features")

            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Remove student_id from imputation
            if 'student_id' in numeric_cols:
                numeric_cols.remove('student_id')
            if 'student_id' in categorical_cols:
                categorical_cols.remove('student_id')

            df_imputed = df.copy()

            # Impute numeric columns
            if numeric_cols:
                self.imputer = SimpleImputer(strategy=self.imputation_strategy)
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
                logger.info(f"Imputed {len(numeric_cols)} numeric features")

            # Impute categorical columns (use most_frequent for categorical)
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])
                logger.info(f"Imputed {len(categorical_cols)} categorical features")

            # Verify no missing values remain
            remaining_missing = df_imputed.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"Still {remaining_missing} missing values after imputation")

            return df_imputed

        except Exception as e:
            logger.error(f"Error during missing value imputation: {e}")
            raise

    def _validate_fused_data(self, df: pd.DataFrame, has_risk_labels: bool):
        """
        Validate the fused dataset for integrity.

        Args:
            df: Fused DataFrame
            has_risk_labels: Whether risk labels were included

        Raises:
            ValueError: If validation fails
        """
        # Check minimum expected columns
        expected_features = [
            'student_id', 'lms_logins_per_week', 'attendance_percentage',
            'late_night_activity_hours', 'gpa', 'gpa_trend', 'course_withdrawals',
            'facility_visits_per_week', 'wifi_hours_per_week'
        ]

        if has_risk_labels:
            expected_features.append('risk_label')

        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Fused data missing expected features: {missing_features}")

        # Check for duplicate student_ids
        duplicate_ids = df['student_id'].duplicated().sum()
        if duplicate_ids > 0:
            raise ValueError(f"Found {duplicate_ids} duplicate student IDs after fusion")

        # Check data types
        numeric_cols = [col for col in df.columns if col != 'student_id']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column '{col}' is not numeric - may need conversion")

        logger.info("Data fusion validation passed")

    def get_fusion_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics of the fused dataset.

        Args:
            df: Fused DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_students': len(df),
            'total_features': len(df.columns) - 1,  # Exclude student_id
            'feature_names': [col for col in df.columns if col != 'student_id'],
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }

        # Add risk distribution if available
        if 'risk_label' in df.columns:
            risk_counts = df['risk_label'].value_counts().to_dict()
            summary['risk_distribution'] = risk_counts
            summary['risk_ratio'] = df['risk_label'].mean()

        return summary

def merge_multimodal_data(behavioral_df: pd.DataFrame,
                         academic_df: pd.DataFrame,
                         engagement_df: pd.DataFrame,
                         risk_df: Optional[pd.DataFrame] = None,
                         imputation_strategy: str = 'mean') -> pd.DataFrame:
    """
    Convenience function for early fusion of multi-modal data.

    Args:
        behavioral_df: Behavioral features DataFrame
        academic_df: Academic features DataFrame
        engagement_df: Engagement features DataFrame
        risk_df: Risk labels DataFrame (optional)
        imputation_strategy: Strategy for missing value imputation

    Returns:
        Fused and imputed DataFrame
    """
    fusion = DataFusion(imputation_strategy)
    fused_df = fusion.merge_multimodal_data(behavioral_df, academic_df, engagement_df, risk_df)
    fused_df = fusion.impute_missing_values(fused_df)

    summary = fusion.get_fusion_summary(fused_df)
    logger.info(f"Fusion summary: {summary}")

    return fused_df
