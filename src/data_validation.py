"""
Data validation and quality checks.
"""

import pandas as pd
import numpy as np
from datetime import datetime

REQUIRED_COLUMNS = [
    'facility_id',
    'day',
    'total_staff',
    'nurses_on_duty',
    'doctors_on_duty',
    'beds_occupied',
    'beds_available',
    'patient_visits',
    'avg_congestion',
    'weighted_demand',
    'staff_to_patient_ratio',
    'beds_occupancy_ratio'
]

COLUMN_RANGES = {
    'total_staff': (0, 200),
    'nurses_on_duty': (0, 150),
    'doctors_on_duty': (0, 50),
    'beds_occupied': (0, 200),
    'beds_available': (0, 200),
    'patient_visits': (0, 1000),
    'avg_congestion': (0, 1),
    'weighted_demand': (0, 200),
    'staff_to_patient_ratio': (0, 10),
    'beds_occupancy_ratio': (0, 1)
}


def validate_input_data(df):
    """
    Validate input data before processing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data to validate
    
    Returns:
    --------
    dict with validation results and errors
    """
    errors = []
    warnings = []
    
    # 1. Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # 2. Check data types
    if 'day' in df.columns:
        try:
            df['day'] = pd.to_datetime(df['day'])
        except:
            errors.append("Column 'day' cannot be converted to datetime")
    
    # 3. Check value ranges
    for col, (min_val, max_val) in COLUMN_RANGES.items():
        if col in df.columns:
            if df[col].min() < min_val or df[col].max() > max_val:
                warnings.append(f"Column '{col}' has values outside expected range [{min_val}, {max_val}]")
    
    # 4. Check for missing values
    missing_counts = df.isnull().sum()
    high_missing = missing_counts[missing_counts > len(df) * 0.1]  # >10% missing
    if not high_missing.empty:
        warnings.append(f"High missing values in: {high_missing.index.tolist()}")
    
    # 5. Check date continuity (per facility)
    if 'facility_id' in df.columns and 'day' in df.columns:
        for facility in df['facility_id'].unique():
            facility_dates = df[df['facility_id'] == facility]['day'].sort_values()
            date_gaps = facility_dates.diff().dt.days
            large_gaps = date_gaps[date_gaps > 3]  # Gaps > 3 days
            if not large_gaps.empty:
                warnings.append(f"Facility {facility} has date gaps > 3 days")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'row_count': len(df),
        'facility_count': df['facility_id'].nunique() if 'facility_id' in df.columns else 0,
        'date_range': {
            'min': df['day'].min().strftime('%Y-%m-%d') if 'day' in df.columns else None,
            'max': df['day'].max().strftime('%Y-%m-%d') if 'day' in df.columns else None
        } if 'day' in df.columns else None
    }


def validate_features(features_df):
    """
    Validate engineered features.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        Data with engineered features
    
    Returns:
    --------
    dict with validation results
    """
    errors = []
    
    # Check required engineered features
    required_features = [
        'staffing_pressure',
        'staff_to_patient_ratio_7d_avg',
        'weighted_demand_7d_avg',
        'beds_occupancy_ratio_7d_avg'
    ]
    
    missing_features = set(required_features) - set(features_df.columns)
    if missing_features:
        errors.append(f"Missing engineered features: {missing_features}")
    
    # Check for infinite values
    inf_cols = features_df.columns[features_df.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        errors.append(f"Infinite values found in: {inf_cols}")
    
    # Check feature correlations (basic sanity check)
    if 'staffing_pressure' in features_df.columns and 'staff_to_patient_ratio' in features_df.columns:
        corr = features_df['staffing_pressure'].corr(features_df['staff_to_patient_ratio'])
        if corr < 0.3:  # Should be positively correlated
            errors.append(f"Low correlation between staffing_pressure and staff_to_patient_ratio: {corr:.2f}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'feature_count': len(features_df.columns),
        'sample_features': features_df.columns.tolist()[:10]  # First 10 features
    }


def check_data_quality_for_prediction(df):
    """
    Check if data is sufficient for predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    
    Returns:
    --------
    dict with quality assessment
    """
    results = {}
    
    for facility_id in df['facility_id'].unique():
        facility_data = df[df['facility_id'] == facility_id]
        
        # Check data recency
        latest_date = facility_data['day'].max()
        days_since_latest = (pd.Timestamp.now() - latest_date).days
        
        # Check data volume
        total_days = len(facility_data)
        
        results[facility_id] = {
            'total_days': total_days,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'days_since_latest': days_since_latest,
            'sufficient_for_prediction': total_days >= 30 and days_since_latest <= 7,
            'recommendation': 'OK' if total_days >= 30 and days_since_latest <= 7 else 'Need more/recent data'
        }
    
    return results


def generate_data_report(df, features_df=None):
    """
    Generate comprehensive data quality report.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw input data
    
    features_df : pandas.DataFrame, optional
        Engineered features
    
    Returns:
    --------
    dict with full report
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_validation': validate_input_data(df)
    }
    
    if features_df is not None:
        report['features_validation'] = validate_features(features_df)
    
    report['prediction_readiness'] = check_data_quality_for_prediction(df)
    
    # Summary statistics
    report['summary'] = {
        'total_rows': len(df),
        'unique_facilities': df['facility_id'].nunique(),
        'date_range': f"{df['day'].min().date()} to {df['day'].max().date()}",
        'total_days': (df['day'].max() - df['day'].min()).days + 1
    }
    
    return report