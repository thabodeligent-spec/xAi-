"""
Data Loader Module for Student Suicide Risk Prediction

This module provides functions to load multi-modal data from various sources:
- Behavioral data (LMS logs, attendance)
- Academic data (GPA, grades)
- Engagement data (facility visits, WiFi usage)

All functions include proper error handling and logging.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading multi-modal student data."""

    def __init__(self, data_dir: str = "data/synthetic"):
        """
        Initialize DataLoader with data directory.

        Args:
            data_dir: Path to the data directory (default: "data/synthetic")
        """
        self.data_dir = Path(data_dir)
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

    def load_behavioral_data(self) -> pd.DataFrame:
        """
        Load behavioral data including LMS logs and attendance.

        Returns:
            DataFrame with behavioral features

        Raises:
            FileNotFoundError: If behavioral_data.csv is not found
            pd.errors.EmptyDataError: If the file is empty
        """
        filepath = self.data_dir / "behavioral_data.csv"

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded behavioral data: {len(df)} records, {len(df.columns)} features")

            # Validate required columns
            required_cols = ['student_id', 'lms_logins_per_week', 'attendance_percentage', 'late_night_activity_hours']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in behavioral data: {missing_cols}")

            return df

        except FileNotFoundError:
            logger.error(f"Behavioral data file not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Behavioral data file is empty: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading behavioral data: {e}")
            raise

    def load_academic_data(self) -> pd.DataFrame:
        """
        Load academic data including GPA and course information.

        Returns:
            DataFrame with academic features

        Raises:
            FileNotFoundError: If academic_data.csv is not found
            pd.errors.EmptyDataError: If the file is empty
        """
        filepath = self.data_dir / "academic_data.csv"

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded academic data: {len(df)} records, {len(df.columns)} features")

            # Validate required columns
            required_cols = ['student_id', 'gpa', 'gpa_trend', 'course_withdrawals']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in academic data: {missing_cols}")

            return df

        except FileNotFoundError:
            logger.error(f"Academic data file not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Academic data file is empty: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading academic data: {e}")
            raise

    def load_engagement_data(self) -> pd.DataFrame:
        """
        Load engagement data including facility visits and WiFi usage.

        Returns:
            DataFrame with engagement features

        Raises:
            FileNotFoundError: If engagement_data.csv is not found
            pd.errors.EmptyDataError: If the file is empty
        """
        filepath = self.data_dir / "engagement_data.csv"

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded engagement data: {len(df)} records, {len(df.columns)} features")

            # Validate required columns
            required_cols = ['student_id', 'facility_visits_per_week', 'wifi_hours_per_week']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in engagement data: {missing_cols}")

            return df

        except FileNotFoundError:
            logger.error(f"Engagement data file not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Engagement data file is empty: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading engagement data: {e}")
            raise

    def load_risk_labels(self) -> pd.DataFrame:
        """
        Load risk labels for supervised learning.

        Returns:
            DataFrame with student_id and risk_label

        Raises:
            FileNotFoundError: If risk_labels.csv is not found
        """
        filepath = self.data_dir / "risk_labels.csv"

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded risk labels: {len(df)} records")

            # Validate required columns
            required_cols = ['student_id', 'risk_label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in risk labels: {missing_cols}")

            return df

        except FileNotFoundError:
            logger.error(f"Risk labels file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading risk labels: {e}")
            raise

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data modalities and risk labels.

        Returns:
            Tuple of (behavioral_df, academic_df, engagement_df, risk_df)

        Raises:
            Exception: If any data loading fails
        """
        try:
            behavioral = self.load_behavioral_data()
            academic = self.load_academic_data()
            engagement = self.load_engagement_data()
            risk = self.load_risk_labels()

            logger.info("Successfully loaded all data modalities")
            return behavioral, academic, engagement, risk

        except Exception as e:
            logger.error(f"Failed to load all data: {e}")
            raise

def load_behavioral_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """
    Convenience function to load behavioral data.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with behavioral features
    """
    loader = DataLoader(data_dir)
    return loader.load_behavioral_data()

def load_academic_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """
    Convenience function to load academic data.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with academic features
    """
    loader = DataLoader(data_dir)
    return loader.load_academic_data()

def load_engagement_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """
    Convenience function to load engagement data.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with engagement features
    """
    loader = DataLoader(data_dir)
    return loader.load_engagement_data()
