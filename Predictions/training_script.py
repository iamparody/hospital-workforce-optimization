# -*- coding: utf-8 -*-
"""Healthcare Staffing Analysis - Training Script"""

import pandas as pd
import numpy as np
import pickle
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA
# ═══════════════════════════════════════════════════════════════

df = pd.read_csv('staffing_context.csv')
df = df.drop(columns="department_name")
df['context_datetime'] = pd.to_datetime(df['context_datetime'], errors='coerce')
df['date'] = pd.to_datetime(df['context_datetime'].dt.date)

# Aggregate to facility-department-date level
agg_df = df.groupby(['facility_id', 'department_id', 'date']).agg({
    'staff_on_duty': 'sum',
    'nurses_on_duty': 'sum',
    'doctors_on_duty': 'sum',
    'beds_occupied': 'sum',
    'beds_available': 'max'
}).reset_index()

# Feature engineering
agg_df['total_beds'] = agg_df['beds_occupied'] + agg_df['beds_available']
agg_df['occupancy_rate'] = np.where(
    agg_df['total_beds'] > 0,
    agg_df['beds_occupied'] / agg_df['total_beds'],
    0.0
)
agg_df['total_staff'] = (
    agg_df['staff_on_duty'] +
    agg_df['nurses_on_duty'] +
    agg_df['doctors_on_duty']
)
agg_df['staff_per_occupied_bed'] = np.where(
    agg_df['beds_occupied'] > 0,
    agg_df['total_staff'] / agg_df['beds_occupied'],
    0.0
)

print(f"✓ Data prepared: {len(agg_df)} records")

# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

OCCUPANCY_HIGH_THRESHOLD = 0.70
MIN_STAFF_PER_OCCUPIED_BED = 0.70
FORECAST_DAYS = 7
OVERLOAD_OCCUPANCY = 0.70
OVERLOAD_STAFF_RATIO = 0.75
MIN_CONSECUTIVE_DAYS = 7

# ═══════════════════════════════════════════════════════════════
# 3. PREPARE DEPARTMENT-LEVEL TIME SERIES
# ═══════════════════════════════════════════════════════════════

dept_daily = agg_df.groupby(['department_id', 'date']).agg({
    'occupancy_rate': 'mean',
    'staff_per_occupied_bed': 'mean',
    'total_staff': 'mean'
}).reset_index()

print(f"✓ Department daily data: {dept_daily.shape}")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN PROPHET MODELS
# ═══════════════════════════════════════════════════════════════

def forecast_department(dept_name, metric='staff_per_occupied_bed', periods=FORECAST_DAYS):
    df_dept = dept_daily[dept_daily['department_id'] == dept_name][['date', metric]].copy()
    df_dept = df_dept.rename(columns={'date': 'ds', metric: 'y'})
    
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive'
    )
    m.fit(df_dept)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_forecast['department_id'] = dept_name
    future_forecast['metric'] = metric
    
    return future_forecast, m

# Train and store models
models = {}
baseline_stats = {}

print("\nTraining models...")
for dept in sorted(dept_daily['department_id'].unique()):
    print(f"  • {dept}")
    
    # Train models
    _, model_staff = forecast_department(dept, 'staff_per_occupied_bed')
    _, model_occ = forecast_department(dept, 'occupancy_rate')
    
    models[f"{dept}_staff"] = model_staff
    models[f"{dept}_occupancy"] = model_occ
    
    # Store baseline (last 30 days)
    hist = dept_daily[dept_daily['department_id'] == dept].tail(30)
    baseline_stats[dept] = {
        'baseline_staff': hist['staff_per_occupied_bed'].mean(),
        'baseline_occ': hist['occupancy_rate'].mean()
    }

print(f"✓ Trained {len(models)} models for {len(baseline_stats)} departments")

# ═══════════════════════════════════════════════════════════════
# 5. SAVE MODEL PACKAGE
# ═══════════════════════════════════════════════════════════════

model_package = {
    'models': models,
    'baseline_stats': baseline_stats,
    'config': {
        'OCCUPANCY_HIGH_THRESHOLD': OCCUPANCY_HIGH_THRESHOLD,
        'MIN_STAFF_PER_OCCUPIED_BED': MIN_STAFF_PER_OCCUPIED_BED,
        'FORECAST_DAYS': FORECAST_DAYS,
        'OVERLOAD_OCCUPANCY': OVERLOAD_OCCUPANCY,
        'OVERLOAD_STAFF_RATIO': OVERLOAD_STAFF_RATIO,
        'MIN_CONSECUTIVE_DAYS': MIN_CONSECUTIVE_DAYS
    }
}

with open('healthcare_models.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✓ Models saved to 'healthcare_models.pkl'")
print("\n" + "="*60)
print("TRAINING COMPLETE - Ready for Streamlit deployment")
print("="*60)
print("\nRun: streamlit run app.py")