"""
Prediction functions for staffing forecasts.
"""

import pickle
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from prophet import Prophet

# Load configuration
CONFIG_PATH = '1_data_generation/saved_models/deployment_config.json'
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        'threshold': 97.03,
        'pressure_to_probability_k': 4
    }


def pressure_to_probability(pressure, threshold, k=4):
    """
    Convert staffing pressure to understaffing probability.
    
    Parameters:
    -----------
    pressure : float
        Staffing pressure value
    
    threshold : float
        Understaffing threshold
    
    k : int
        Steepness parameter (higher = sharper transition)
    
    Returns:
    --------
    float probability between 0 and 1
    """
    x = pressure / threshold
    return 1 / (1 + np.exp(-k * (x - 1)))


def load_facility_model(facility_id):
    """
    Load a saved Prophet model for a facility.
    
    Parameters:
    -----------
    facility_id : str
        Facility identifier
    
    Returns:
    --------
    Prophet model object
    """
    model_path = f'1_data_generation/saved_models/prophet_model_{facility_id}.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def predict_single_facility(facility_data, days_ahead=7):
    """
    Predict understaffing for a single facility.
    
    Parameters:
    -----------
    facility_data : pandas.DataFrame
        Facility data with engineered features
    
    days_ahead : int
        Days to forecast (default: 7)
    
    Returns:
    --------
    DataFrame with predictions
    """
    facility_id = facility_data['facility_id'].iloc[0]
    
    try:
        # Load model
        model = load_facility_model(facility_id)
        
        # Prepare data for Prophet
        prophet_df = facility_data[['day', 'staffing_pressure', 'weighted_demand']].copy()
        prophet_df.columns = ['ds', 'y', 'weighted_demand']
        prophet_df = prophet_df.sort_values('ds')
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        
        # Add regressor values (use last 7-day average)
        last_demand = facility_data['weighted_demand'].iloc[-7:].mean()
        future['weighted_demand'] = last_demand
        
        # Make forecast
        forecast = model.predict(future)
        future_forecast = forecast.tail(days_ahead).copy()
        
        # Calculate probabilities
        threshold = CONFIG.get('threshold', 97.03)
        future_forecast['understaffing_probability'] = future_forecast['yhat'].apply(
            lambda x: pressure_to_probability(x, threshold, CONFIG.get('pressure_to_probability_k', 4))
        )
        
        future_forecast['predicted_understaffed'] = (
            future_forecast['understaffing_probability'] > 0.5
        )
        
        # Add facility info
        future_forecast['facility_id'] = facility_id
        
        return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 
                               'understaffing_probability', 'predicted_understaffed', 'facility_id']]
        
    except Exception as e:
        print(f"Error predicting for {facility_id}: {str(e)}")
        return pd.DataFrame()


def predict_next_7_days(full_data, feature_engineered=True):
    """
    Main prediction function for all facilities.
    
    Parameters:
    -----------
    full_data : pandas.DataFrame
        Data for all facilities
    
    feature_engineered : bool
        Whether features are already engineered (default: True)
    
    Returns:
    --------
    DataFrame with predictions for all facilities
    """
    from feature_engineering import create_features
    
    # Engineer features if needed
    if not feature_engineered:
        full_data = create_features(full_data)
    
    # Get unique facilities
    facilities = full_data['facility_id'].unique()
    
    all_predictions = []
    
    for facility_id in facilities:
        facility_data = full_data[full_data['facility_id'] == facility_id].copy()
        
        # Need at least 30 days of data
        if len(facility_data) < 30:
            print(f"Skipping {facility_id}: insufficient data ({len(facility_data)} days)")
            continue
        
        # Make predictions
        predictions = predict_single_facility(facility_data, days_ahead=7)
        
        if not predictions.empty:
            all_predictions.append(predictions)
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return pd.DataFrame()


def generate_executive_summary(predictions_df):
    """
    Create executive summary from predictions.
    
    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        Predictions from predict_next_7_days()
    
    Returns:
    --------
    dict with executive summary
    """
    if predictions_df.empty:
        return {"message": "No predictions available"}
    
    summary = predictions_df.groupby('facility_id').agg({
        'understaffing_probability': 'max',
        'predicted_understaffed': 'any'
    }).reset_index()
    
    summary.columns = ['facility_id', 'max_risk_next_7d', 'understaffed_any_day']
    
    # Add risk levels
    summary['risk_level'] = pd.cut(
        summary['max_risk_next_7d'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True
    )
    
    # Counts
    risk_counts = summary['risk_level'].value_counts().to_dict()
    
    # Facilities to watch
    watchlist = summary[
        summary['max_risk_next_7d'] > 0.4
    ]['facility_id'].tolist()
    
    return {
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_facilities': len(summary),
        'risk_distribution': risk_counts,
        'high_risk_facilities': risk_counts.get('High', 0) + risk_counts.get('Critical', 0),
        'watchlist': watchlist,
        'facility_details': summary.to_dict('records')
    }


def save_predictions(predictions_df, output_dir='predictions'):
    """
    Save predictions to CSV files.
    
    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        Predictions to save
    
    output_dir : str
        Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    detailed_path = f'{output_dir}/detailed_predictions_{timestamp}.csv'
    predictions_df.to_csv(detailed_path, index=False)
    
    # Generate and save summary
    summary = generate_executive_summary(predictions_df)
    summary_path = f'{output_dir}/executive_summary_{timestamp}.json'
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Predictions saved to: {detailed_path}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return detailed_path, summary_path