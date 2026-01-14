"""
Feature engineering pipeline for staffing predictions.
Converts raw hospital metrics into model-ready features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_features(df, lookback_days=7):
    """
    Create all engineered features from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data with columns: facility_id, day, staff_to_patient_ratio,
        weighted_demand, beds_occupancy_ratio, patient_visits, avg_congestion
    
    lookback_days : int
        Window for rolling calculations (default: 7 days)
    
    Returns:
    --------
    pandas.DataFrame with engineered features
    """
    
    # Convert day to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['day']):
        df['day'] = pd.to_datetime(df['day'])
    
    df = df.sort_values(['facility_id', 'day'])
    
    # 1. Create staffing pressure (target variable)
    df['staffing_pressure'] = df['staff_to_patient_ratio'] * df['weighted_demand']
    
    # Initialize results list
    results = []
    
    # Process each facility separately
    for facility_id in df['facility_id'].unique():
        facility_df = df[df['facility_id'] == facility_id].copy()
        
        # 2. Rolling averages
        for col in ['staff_to_patient_ratio', 'weighted_demand', 
                   'beds_occupancy_ratio', 'patient_visits', 'avg_congestion']:
            facility_df[f'{col}_7d_avg'] = facility_df[col].rolling(
                window=lookback_days, min_periods=3
            ).mean()
        
        # 3. Trend features (difference in rolling averages)
        facility_df['staff_ratio_trend'] = facility_df['staff_to_patient_ratio_7d_avg'].diff(3)
        facility_df['demand_trend'] = facility_df['weighted_demand_7d_avg'].diff(3)
        facility_df['occupancy_trend'] = facility_df['beds_occupancy_ratio_7d_avg'].diff(3)
        
        # 4. Current vs average
        facility_df['ratio_vs_avg'] = (
            facility_df['staff_to_patient_ratio'] / facility_df['staff_to_patient_ratio_7d_avg']
        )
        
        # 5. Temporal features
        facility_df['day_of_week'] = facility_df['day'].dt.dayofweek
        facility_df['is_weekend'] = facility_df['day_of_week'].isin([5, 6]).astype(int)
        facility_df['month'] = facility_df['day'].dt.month
        
        # 6. Rolling staffing pressure
        facility_df['pressure_7d_avg'] = facility_df['staffing_pressure'].rolling(
            window=lookback_days, min_periods=3
        ).mean()
        
        results.append(facility_df)
    
    # Combine all facilities
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df


def prepare_for_prediction(facility_data, days_forward=7):
    """
    Prepare the latest data for forecasting.
    
    Parameters:
    -----------
    facility_data : pandas.DataFrame
        Single facility's data with engineered features
    
    days_forward : int
        Number of days to forecast (default: 7)
    
    Returns:
    --------
    dict with prepared data and future dates
    """
    # Get the last row (most recent data)
    latest = facility_data.iloc[-1].copy()
    
    # Prepare future dates
    last_date = pd.to_datetime(latest['day'])
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_forward + 1)]
    
    # Calculate future weighted_demand (simple: use last 7-day average)
    last_7_demand = facility_data['weighted_demand'].iloc[-7:].mean()
    
    prepared_data = {
        'facility_id': latest['facility_id'],
        'last_date': last_date,
        'last_pressure': latest['staffing_pressure'],
        'last_7d_avg_pressure': latest['pressure_7d_avg'],
        'future_dates': future_dates,
        'future_demand': last_7_demand,  # For Prophet regressor
        'current_features': {
            'staff_ratio_7d_avg': latest['staff_to_patient_ratio_7d_avg'],
            'demand_7d_avg': latest['weighted_demand_7d_avg'],
            'occupancy_7d_avg': latest['beds_occupancy_ratio_7d_avg'],
            'staff_ratio_trend': latest['staff_ratio_trend'],
            'demand_trend': latest['demand_trend'],
            'day_of_week': last_date.dayofweek
        }
    }
    
    return prepared_data


def calculate_threshold(df, percentile=0.6):
    """
    Calculate understaffing threshold from historical data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with staffing_pressure column
    
    percentile : float
        Percentile to use as threshold (default: 0.6 = 60th)
    
    Returns:
    --------
    float threshold value
    """
    threshold = df['staffing_pressure'].quantile(percentile)
    
    # Calculate equivalent ratio threshold
    avg_demand = df['weighted_demand'].mean()
    equivalent_ratio = threshold / avg_demand
    
    return {
        'pressure_threshold': float(threshold),
        'equivalent_ratio_threshold': float(equivalent_ratio),
        'avg_demand': float(avg_demand),
        'percentile_used': percentile
    }