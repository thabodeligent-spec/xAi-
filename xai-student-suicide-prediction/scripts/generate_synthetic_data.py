#!/usr/bin/env python3
"""
Synthetic Data Generator for Student Suicide Risk Prediction

This script generates synthetic multi-modal data for 1000 students including:
- Behavioral features (LMS logs, attendance, late-night activity)
- Academic features (GPA, GPA trend, course withdrawals)
- Engagement features (facility visits, WiFi hours)

Risk labels are generated based on patterns from literature review.
Class imbalance: ~5% high-risk students.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
N_STUDENTS = 1000
RISK_RATIO = 0.05  # 5% high-risk students
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_student_ids(n_students: int) -> list:
    """Generate unique student IDs."""
    return [f"STU{i:04d}" for i in range(1, n_students + 1)]

def generate_behavioral_data(student_ids: list) -> pd.DataFrame:
    """
    Generate behavioral features:
    - lms_logins: 0-100 per week
    - attendance_percentage: 0-100%
    - late_night_activity: 0-20 hours per week
    """
    logger.info("Generating behavioral data...")

    data = []
    for student_id in student_ids:
        # Base distributions
        lms_logins = np.random.normal(50, 20)  # Mean 50, std 20
        attendance = np.random.normal(85, 15)  # Mean 85%, std 15
        late_night = np.random.normal(5, 3)    # Mean 5 hours, std 3

        # Clip to valid ranges
        lms_logins = np.clip(lms_logins, 0, 100)
        attendance = np.clip(attendance, 0, 100)
        late_night = np.clip(late_night, 0, 20)

        data.append({
            'student_id': student_id,
            'lms_logins_per_week': round(lms_logins, 1),
            'attendance_percentage': round(attendance, 1),
            'late_night_activity_hours': round(late_night, 1)
        })

    return pd.DataFrame(data)

def generate_academic_data(student_ids: list) -> pd.DataFrame:
    """
    Generate academic features:
    - gpa: 0-4.0
    - gpa_trend: -2 to +2 (change over semester)
    - course_withdrawals: 0-3
    """
    logger.info("Generating academic data...")

    data = []
    for student_id in student_ids:
        # GPA distribution (slightly skewed toward higher values)
        gpa = np.random.beta(7, 3) * 4.0  # Beta distribution for realistic GPA

        # GPA trend (most students stable, some declining)
        gpa_trend = np.random.normal(0, 0.5)  # Mean 0, std 0.5
        gpa_trend = np.clip(gpa_trend, -2, 2)

        # Course withdrawals (Poisson distribution)
        withdrawals = np.random.poisson(0.3)  # Mean 0.3 withdrawals
        withdrawals = min(withdrawals, 3)  # Cap at 3

        data.append({
            'student_id': student_id,
            'gpa': round(gpa, 2),
            'gpa_trend': round(gpa_trend, 2),
            'course_withdrawals': int(withdrawals)
        })

    return pd.DataFrame(data)

def generate_engagement_data(student_ids: list) -> pd.DataFrame:
    """
    Generate engagement features:
    - facility_visits_per_week: 0-30
    - wifi_hours_per_week: 0-80
    """
    logger.info("Generating engagement data...")

    data = []
    for student_id in student_ids:
        # Facility visits (library, gym, etc.)
        facility_visits = np.random.normal(15, 8)  # Mean 15, std 8
        facility_visits = np.clip(facility_visits, 0, 30)

        # WiFi hours (study time, online activity)
        wifi_hours = np.random.normal(40, 15)  # Mean 40, std 15
        wifi_hours = np.clip(wifi_hours, 0, 80)

        data.append({
            'student_id': student_id,
            'facility_visits_per_week': round(facility_visits, 1),
            'wifi_hours_per_week': round(wifi_hours, 1)
        })

    return pd.DataFrame(data)

def assign_risk_labels(behavioral_df: pd.DataFrame,
                      academic_df: pd.DataFrame,
                      engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk labels based on patterns from literature:
    HIGH RISK if:
    - (GPA_trend < -0.5 AND attendance < 60) OR
    - (LMS_logins < 10 AND GPA < 2.0)

    Target: ~5% high-risk students
    """
    logger.info("Assigning risk labels...")

    # Merge data for risk calculation
    merged = behavioral_df.merge(academic_df, on='student_id')
    merged = merged.merge(engagement_df, on='student_id')

    # Risk criteria
    risk_condition1 = (merged['gpa_trend'] < -0.5) & (merged['attendance_percentage'] < 60)
    risk_condition2 = (merged['lms_logins_per_week'] < 10) & (merged['gpa'] < 2.0)

    # Assign risk (1 = high risk, 0 = low risk)
    merged['risk_label'] = (risk_condition1 | risk_condition2).astype(int)

    # Adjust to achieve target risk ratio
    current_risk_ratio = merged['risk_label'].mean()
    if current_risk_ratio > RISK_RATIO:
        # Too many high-risk, randomly set some to low-risk
        high_risk_indices = merged[merged['risk_label'] == 1].index
        n_to_flip = int((current_risk_ratio - RISK_RATIO) * len(merged))
        flip_indices = np.random.choice(high_risk_indices, n_to_flip, replace=False)
        merged.loc[flip_indices, 'risk_label'] = 0
    elif current_risk_ratio < RISK_RATIO:
        # Too few high-risk, randomly set some to high-risk
        low_risk_indices = merged[merged['risk_label'] == 0].index
        n_to_flip = int((RISK_RATIO - current_risk_ratio) * len(merged))
        flip_indices = np.random.choice(low_risk_indices, n_to_flip, replace=False)
        merged.loc[flip_indices, 'risk_label'] = 1

    final_risk_ratio = merged['risk_label'].mean()
    logger.info(".3f")

    return merged[['student_id', 'risk_label']]

def save_dataframes(dataframes: dict, output_dir: Path):
    """Save dataframes to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        filename = f"{name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {filename} with {len(df)} records")

def main():
    """Main function to generate and save synthetic data."""
    logger.info("Starting synthetic data generation...")

    # Create output directory
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate student IDs
    student_ids = generate_student_ids(N_STUDENTS)

    # Generate each modality
    behavioral_df = generate_behavioral_data(student_ids)
    academic_df = generate_academic_data(student_ids)
    engagement_df = generate_engagement_data(student_ids)

    # Assign risk labels
    risk_df = assign_risk_labels(behavioral_df, academic_df, engagement_df)

    # Save dataframes
    dataframes = {
        'behavioral_data': behavioral_df,
        'academic_data': academic_df,
        'engagement_data': engagement_df,
        'risk_labels': risk_df
    }

    save_dataframes(dataframes, output_dir)

    # Print summary statistics
    logger.info("Data generation complete!")
    logger.info(f"Total students: {N_STUDENTS}")
    logger.info(".3f")
    logger.info(f"High-risk students: {risk_df['risk_label'].sum()}")

    # Display sample of each dataset
    logger.info("\nBehavioral data sample:")
    logger.info(behavioral_df.head().to_string())

    logger.info("\nAcademic data sample:")
    logger.info(academic_df.head().to_string())

    logger.info("\nEngagement data sample:")
    logger.info(engagement_df.head().to_string())

    logger.info("\nRisk labels sample:")
    logger.info(risk_df.head().to_string())

if __name__ == "__main__":
    main()
