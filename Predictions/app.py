# ────────────────────────────────────────────────────────────────
# app.py - Streamlit Application
# ────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Healthcare Staffing Analytics", layout="wide")

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    try:
        with open('healthcare_models.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Run training_script.py first.")
        st.stop()

model_package = load_models()
models = model_package['models']
baseline_stats = model_package['baseline_stats']
config = model_package['config']

# ═══════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess_raw_data(df):
    """Convert raw CSV to aggregated format"""
    # Handle date column
    if 'context_datetime' in df.columns:
        df['context_datetime'] = pd.to_datetime(df['context_datetime'])
        df['date'] = pd.to_datetime(df['context_datetime'].dt.date)
    elif 'date' not in df.columns:
        st.error("❌ CSV must contain 'date' or 'context_datetime' column")
        st.stop()
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    # Check if already aggregated
    required_cols = ['occupancy_rate', 'staff_per_occupied_bed', 'total_staff']
    if all(col in df.columns for col in required_cols):
        return df  # Already processed
    
    # Aggregate raw data
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
    
    return agg_df

# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def predict_understaffing(agg_df):
    """Q1: Predict understaffing risk for next 7 days"""
    dept_daily = agg_df.groupby(['department_id', 'date']).agg({
        'occupancy_rate': 'mean',
        'staff_per_occupied_bed': 'mean',
        'total_staff': 'mean'
    }).reset_index()
    
    risk_scores = []
    for dept in baseline_stats.keys():
        staff_model = models[f"{dept}_staff"]
        occ_model = models[f"{dept}_occupancy"]
        
        future = staff_model.make_future_dataframe(periods=config['FORECAST_DAYS'])
        
        staff_fc = staff_model.predict(future).tail(config['FORECAST_DAYS'])['yhat'].values
        occ_fc = occ_model.predict(future).tail(config['FORECAST_DAYS'])['yhat'].values
        
        baseline_staff = baseline_stats[dept]['baseline_staff']
        baseline_occ = baseline_stats[dept]['baseline_occ']
        
        occ_change = ((occ_fc.mean() - baseline_occ) / baseline_occ) * 100
        staff_change = ((staff_fc.mean() - baseline_staff) / baseline_staff) * 100
        risk_score = (occ_change * 0.6) - (staff_change * 0.4)
        
        understaff_days = ((occ_fc > baseline_occ) & (staff_fc < baseline_staff)).sum()
        
        risk_scores.append({
            'department_id': dept,
            'baseline_occupancy': round(baseline_occ, 3),
            'forecast_occupancy': round(occ_fc.mean(), 3),
            'occ_change_%': round(occ_change, 2),
            'baseline_staff_ratio': round(baseline_staff, 3),
            'forecast_staff_ratio': round(staff_fc.mean(), 3),
            'staff_change_%': round(staff_change, 2),
            'days_worse_than_baseline': int(understaff_days),
            'risk_score': round(risk_score, 2)
        })
    
    return pd.DataFrame(risk_scores).sort_values('risk_score', ascending=False)

def analyze_sustained_overload(agg_df):
    """Q2: Detect sustained overload patterns"""
    agg_df = agg_df.copy()
    agg_df['is_overloaded'] = (
        (agg_df['occupancy_rate'] > config['OVERLOAD_OCCUPANCY']) &
        (agg_df['staff_per_occupied_bed'] < config['OVERLOAD_STAFF_RATIO'])
    ).astype(int)
    
    agg_df_sorted = agg_df.sort_values(['department_id', 'facility_id', 'date'])
    agg_df_sorted['overload_streak'] = (
        agg_df_sorted.groupby(['department_id', 'facility_id'])['is_overloaded']
        .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumsum()))
    )
    
    overload_df = agg_df_sorted.groupby('department_id').agg({
        'overload_streak': 'max',
        'is_overloaded': 'sum',
        'date': 'count'
    }).reset_index()
    
    overload_df.columns = ['department_id', 'max_consecutive_days', 'total_overload_days', 'total_days']
    overload_df['overload_%'] = (overload_df['total_overload_days'] / overload_df['total_days'] * 100).round(2)
    
    return overload_df.sort_values('max_consecutive_days', ascending=False)

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

st.title("🏥 Healthcare Staffing Analytics Dashboard")
st.markdown("Upload healthcare data to predict understaffing risk and detect sustained overload")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(f"""
    **Model Configuration:**
    - Forecast period: {config['FORECAST_DAYS']} days
    - High occupancy: > {config['OCCUPANCY_HIGH_THRESHOLD']}
    - Low staff ratio: < {config['MIN_STAFF_PER_OCCUPIED_BED']}
    - Overload threshold: {config['MIN_CONSECUTIVE_DAYS']} consecutive days
    
    **Departments tracked:**
    {', '.join(baseline_stats.keys())}
    """)

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload either raw staffing_context.csv or preprocessed agg_df.csv"
)

if uploaded_file:
    # Load and preprocess
    with st.spinner("Loading data..."):
        raw_df = pd.read_csv(uploaded_file)
        agg_df = preprocess_raw_data(raw_df)
    
    st.success(f"✓ Loaded {len(agg_df):,} records from {agg_df['date'].min().date()} to {agg_df['date'].max().date()}")
    
    # Generate predictions
    with st.spinner("Generating predictions..."):
        risk_df = predict_understaffing(agg_df)
        overload_df = analyze_sustained_overload(agg_df)
    
    # Display results
    tab1, tab2, tab3 = st.tabs([
        "📊 Q1: Understaffing Risk",
        "⚠️ Q2: Sustained Overload",
        "📋 Summary"
    ])
    
    # ─────────────────────────────────────────────────────────
    # TAB 1: Understaffing Risk
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Departments at Risk (Next 7 Days)")
        
        high_risk = risk_df[risk_df['risk_score'] > 0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk Departments", len(high_risk), 
                   delta=f"{len(high_risk)}/{len(risk_df)}")
        col2.metric("Highest Risk Score", f"{risk_df['risk_score'].max():.2f}")
        col3.metric("Average Risk", f"{risk_df['risk_score'].mean():.2f}")
        
        # Risk chart
        fig = px.bar(
            risk_df,
            x='department_id',
            y='risk_score',
            color='risk_score',
            color_continuous_scale='RdYlGn_r',
            title="Risk Score by Department",
            labels={'risk_score': 'Risk Score', 'department_id': 'Department'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(
            risk_df,
            use_container_width=True,
            column_config={
                "risk_score": st.column_config.NumberColumn(
                    "Risk Score",
                    help="Positive = deteriorating conditions",
                    format="%.2f"
                )
            }
        )
    
    # ─────────────────────────────────────────────────────────
    # TAB 2: Sustained Overload
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Historical Sustained Overload Analysis")
        
        sustained = overload_df[overload_df['max_consecutive_days'] >= config['MIN_CONSECUTIVE_DAYS']]
        
        col1, col2 = st.columns(2)
        col1.metric("Sustained Overload Departments", len(sustained))
        col2.metric("Longest Streak", f"{overload_df['max_consecutive_days'].max()} days")
        
        # Overload chart
        fig = px.bar(
            overload_df,
            x='department_id',
            y='max_consecutive_days',
            color='overload_%',
            color_continuous_scale='Reds',
            title="Maximum Consecutive Overload Days",
            labels={'max_consecutive_days': 'Consecutive Days', 'department_id': 'Department'}
        )
        fig.add_hline(
            y=config['MIN_CONSECUTIVE_DAYS'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Sustained threshold ({config['MIN_CONSECUTIVE_DAYS']} days)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(overload_df, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────
    # TAB 3: Executive Summary
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Executive Summary")
        
        st.markdown("### 📊 Q1: Understaffing Risk (Next 7 Days)")
        if len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                with st.expander(f"🔴 **{row['department_id']}** - Risk Score: {row['risk_score']:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Occupancy Change",
                            f"{row['forecast_occupancy']:.3f}",
                            delta=f"{row['occ_change_%']:+.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Staff Ratio Change",
                            f"{row['forecast_staff_ratio']:.3f}",
                            delta=f"{row['staff_change_%']:+.1f}%",
                            delta_color="inverse"
                        )
                    st.caption(f"Days worse than baseline: {row['days_worse_than_baseline']}")
        else:
            st.success("✅ No departments at risk - all showing stable/improving conditions")
        
        st.markdown("### ⚠️ Q2: Sustained Overload")
        if len(sustained) > 0:
            for _, row in sustained.iterrows():
                with st.expander(f"🔴 **{row['department_id']}**"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Longest Streak", f"{row['max_consecutive_days']} days")
                    col2.metric("Total Overload Days", row['total_overload_days'])
                    col3.metric("Overload %", f"{row['overload_%']}%")
        else:
            st.success("✅ No sustained overload detected")

else:
    st.info("👆 Upload a CSV file to begin analysis")
    
    # Sample data format
    with st.expander("📄 Expected CSV Format"):
        st.markdown("""
        **Option 1: Raw data** (will be auto-processed)
        - `context_datetime`, `facility_id`, `department_id`
        - `staff_on_duty`, `nurses_on_duty`, `doctors_on_duty`
        - `beds_occupied`, `beds_available`
        
        **Option 2: Pre-aggregated data**
        - `date`, `facility_id`, `department_id`
        - `occupancy_rate`, `staff_per_occupied_bed`, `total_staff`
        """)