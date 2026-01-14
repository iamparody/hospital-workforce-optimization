# predict_staffing.py
import pandas as pd
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet

print("🏥 HOSPITAL STAFFING PREDICTOR")
print("=" * 50)

# ========== 1. LOAD MODELS & CONFIG ==========
print("\n📂 Loading models...")

# Load config
config_path = '1_data_generation/saved_models/deployment_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"✓ Config loaded: threshold = {config['threshold']}")

# Load one model to test
model_path = '1_data_generation/saved_models/prophet_model_HF_L4_004.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"✓ Model loaded: prophet_model_HF_L4_004.pkl")

# ========== 2. CREATE TEST DATA ==========
print("\n📊 Creating test data for HF_L4_004...")

# Create 7 days of recent data (minimum needed for 7-day rolling features)
test_dates = [datetime(2024, 12, 20) + timedelta(days=i) for i in range(7)]
test_data = []

for i, date in enumerate(test_dates):
    # Realistic values based on your data patterns
    test_data.append({
        'facility_id': 'HF_L4_004',
        'day': date.strftime('%Y-%m-%d'),
        'total_staff': 45,
        'nurses_on_duty': 30,
        'doctors_on_duty': 10,
        'beds_occupied': 28 + i,  # Increasing slightly
        'beds_available': 2,
        'patient_visits': 85.0 + i*2,
        'avg_congestion': 0.65 + i*0.02,
        'weighted_demand': 52.3 + i*0.5,  # From your config avg: 49.2
        'staff_to_patient_ratio': 0.35 + i*0.01,  # Increasing = worse
        'beds_occupancy_ratio': 0.93
    })

test_df = pd.DataFrame(test_data)
print(f"Created {len(test_df)} days of test data")

# ========== 3. ENGINEER FEATURES ==========
print("\n⚙️ Engineering features...")

# Convert day to datetime
test_df['day'] = pd.to_datetime(test_df['day'])
test_df = test_df.sort_values(['facility_id', 'day'])

# Calculate staffing pressure (model target)
test_df['staffing_pressure'] = (
    test_df['staff_to_patient_ratio'] * test_df['weighted_demand']
)

print(f"Staffing pressure range: {test_df['staffing_pressure'].min():.1f} to {test_df['staffing_pressure'].max():.1f}")

# ========== 4. PREPARE FOR PROPHET ==========
print("\n🔮 Preparing for prediction...")

# Prepare data in Prophet format
prophet_df = test_df[['day', 'staffing_pressure', 'weighted_demand']].copy()
prophet_df.columns = ['ds', 'y', 'weighted_demand']
prophet_df = prophet_df.sort_values('ds')

print(f"Prophet data shape: {prophet_df.shape}")

# ========== 5. MAKE PREDICTION ==========
print("\n📈 Making 7-day forecast...")

# Create future dates
future = model.make_future_dataframe(periods=7, freq='D')

# Add regressor values (use last 7-day average)
last_demand = test_df['weighted_demand'].iloc[-7:].mean()
future['weighted_demand'] = last_demand

# Predict
forecast = model.predict(future)
next_7_days = forecast.tail(7).copy()

print(f"Forecast dates: {next_7_days['ds'].min().date()} to {next_7_days['ds'].max().date()}")

# ========== 6. CALCULATE UNDERSTAFFING PROBABILITY ==========
print("\n🎯 Calculating understaffing risk...")

threshold = config['threshold']

def pressure_to_probability(pressure, threshold, k=4):
    """Convert pressure to probability"""
    x = pressure / threshold
    return 1 / (1 + np.exp(-k * (x - 1)))

# Add probabilities
next_7_days['understaffing_probability'] = next_7_days['yhat'].apply(
    lambda x: pressure_to_probability(x, threshold)
)
next_7_days['predicted_understaffed'] = next_7_days['understaffing_probability'] > 0.5

# ========== 7. DISPLAY RESULTS ==========
print("\n" + "=" * 50)
print("PREDICTION RESULTS: HF_L4_004")
print("=" * 50)

print(f"\n📅 Next 7 days forecast (starting {next_7_days['ds'].iloc[0].date()}):")
print("-" * 80)

for _, row in next_7_days.iterrows():
    date_str = row['ds'].strftime('%a %b %d')
    pressure = row['yhat']
    prob = row['understaffing_probability']
    understaffed = row['predicted_understaffed']
    
    risk_level = "🟢 Low" if prob < 0.3 else \
                 "🟡 Medium" if prob < 0.5 else \
                 "🟠 High" if prob < 0.7 else \
                 "🔴 Critical"
    
    status = "🚨 UNDERSTAFFED" if understaffed else "✅ Adequate"
    
    print(f"{date_str}: {pressure:.1f} pressure | {prob:.1%} risk | {risk_level} | {status}")

# Summary
any_understaffed = next_7_days['predicted_understaffed'].any()
max_risk = next_7_days['understaffing_probability'].max()
risk_days = next_7_days['predicted_understaffed'].sum()

print("\n" + "=" * 50)
print("EXECUTIVE SUMMARY")
print("=" * 50)
print(f"Facility: HF_L4_004")
print(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Forecast Period: Next 7 days")
print(f"Max Risk: {max_risk:.1%}")
print(f"Understaffed Days: {risk_days} of 7")
print(f"Overall Status: {'🚨 NEEDS ATTENTION' if any_understaffed else '✅ ALL CLEAR'}")

if any_understaffed:
    risky_dates = next_7_days[next_7_days['predicted_understaffed']]['ds'].dt.strftime('%b %d').tolist()
    print(f"Risky Dates: {', '.join(risky_dates)}")
    
print("\n" + "=" * 50)
print("ACTION RECOMMENDATIONS")
print("=" * 50)

if max_risk > 0.6:
    print("1. 🚨 IMMEDIATE ACTION: Schedule additional staff")
    print("2. 📊 Review patient volume forecasts")
    print("3. 👥 Contact float pool/nursing agency")
elif max_risk > 0.4:
    print("1. ⚠️ MONITOR CLOSELY: Watch staffing ratios daily")
    print("2. 📈 Check trend direction (improving/worsening)")
    print("3. 📋 Prepare contingency plans")
else:
    print("1. ✅ ROUTINE MONITORING: Current staffing adequate")
    print("2. 📊 Continue daily checks")
    print("3. 💡 Optimize staff allocation if needed")