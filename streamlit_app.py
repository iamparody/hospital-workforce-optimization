import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import json
import pickle
import tempfile

# Add src to path
sys.path.append('workforce/src')

# Import modules
try:
    from feature_engineering import create_features
    from model_predict import predict_next_7_days, generate_executive_summary
    from readmission_predictor import ReadmissionPredictor
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all modules are in workforce/src/")

# Set page config
st.set_page_config(
    page_title="Workforce Optimization Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #059669; font-weight: bold; }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">🏥 Workforce Optimization Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive analytics for staffing, readmissions, and facility capacity management</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Staffing Shortage Prediction",
    "🔄 Patient Readmission Risk", 
    "⚠️ Facility Overload Detection"
])

# ============================================================================
# TAB 1: STAFFING SHORTAGE PREDICTION
# ============================================================================
with tab1:
    st.header("📊 Staffing Shortage Prediction")
    st.markdown("Predict staffing shortages 7 days in advance for proactive resource allocation")
    
    # Initialize session state for tab 1
    if 'staffing_predictions' not in st.session_state:
        st.session_state.staffing_predictions = None
    if 'staffing_data' not in st.session_state:
        st.session_state.staffing_data = None
    if 'staffing_features' not in st.session_state:
        st.session_state.staffing_features = None
    
    # Sidebar for Tab 1
    with st.sidebar:
        st.subheader("📤 Staffing Data Input")
        
        staffing_file = st.file_uploader(
            "Upload daily metrics CSV",
            type=['csv'],
            key='staffing_upload',
            help="Upload CSV with columns: facility_id, day, total_staff, nurses_on_duty, doctors_on_duty, beds_occupied, beds_available, patient_visits, avg_congestion, weighted_demand, staff_to_patient_ratio, beds_occupancy_ratio"
        )
        
        st.divider()
        
        st.subheader("⚙️ Prediction Settings")
        staffing_threshold = st.slider(
            "Risk threshold (%)",
            min_value=30,
            max_value=70,
            value=50,
            key='staffing_threshold',
            help="Probability above which facility is considered 'at risk'"
        )
        
        staffing_days = st.slider(
            "Forecast horizon (days)",
            min_value=3,
            max_value=14,
            value=7,
            key='staffing_days',
            help="Number of days to predict ahead"
        )
        
        staffing_predict_btn = st.button(
            "🚀 Run Staffing Predictions",
            type="primary",
            key='staffing_predict',
            use_container_width=True
        )
    
    # Load data
    if staffing_file is not None:
        try:
            df_staffing = pd.read_csv(staffing_file)
            st.session_state.staffing_data = df_staffing
            st.success(f"✅ Data loaded: {len(df_staffing)} rows, {df_staffing['facility_id'].nunique()} facilities")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Run predictions
    if staffing_predict_btn and st.session_state.staffing_data is not None:
        with st.spinner("Running staffing predictions..."):
            try:
                # Engineer features if needed
                if st.session_state.staffing_features is None:
                    features_df = create_features(st.session_state.staffing_data)
                    st.session_state.staffing_features = features_df
                
                # Make predictions
                predictions = predict_next_7_days(st.session_state.staffing_features, feature_engineered=True)
                
                # Apply custom threshold
                predictions['predicted_understaffed'] = predictions['understaffing_probability'] > (staffing_threshold / 100)
                
                st.session_state.staffing_predictions = predictions
                st.success(f"✅ Predictions generated for {predictions['facility_id'].nunique()} facilities")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.error("Make sure models are in 'workforce/1_data_generation/saved_models/' directory")
    
    # Display results
    if st.session_state.staffing_predictions is not None:
        predictions = st.session_state.staffing_predictions
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Facilities Monitored", predictions['facility_id'].nunique())
        
        with col2:
            at_risk = predictions[predictions['predicted_understaffed']]['facility_id'].nunique()
            st.metric("At Risk Facilities", at_risk)
        
        with col3:
            total_days = len(predictions)
            risk_days = predictions['predicted_understaffed'].sum()
            st.metric("Risk Days Next Week", f"{risk_days}/{total_days}")
        
        # Risk heatmap
        st.subheader("📅 7-Day Risk Forecast")
        
        heatmap_data = predictions.pivot_table(
            index='facility_id',
            columns='ds',
            values='understaffing_probability',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            labels=dict(color="Risk %"),
            title="Understaffing Risk by Facility and Day"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Facility selector for detailed view
        st.subheader("🔍 Detailed Facility Forecast")
        
        facilities = predictions['facility_id'].unique()
        selected_facility = st.selectbox("Select Facility", facilities, key='staffing_facility_select')
        
        facility_preds = predictions[predictions['facility_id'] == selected_facility].copy()
        facility_preds['Risk %'] = (facility_preds['understaffing_probability'] * 100).round(1)
        
        # Plot forecast
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=facility_preds['ds'],
            y=facility_preds['yhat'],
            mode='lines+markers',
            name='Predicted Pressure',
            line=dict(color='#3B82F6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.concat([facility_preds['ds'], facility_preds['ds'][::-1]]),
            y=pd.concat([facility_preds['yhat_upper'], facility_preds['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.add_hline(y=97.03, line_dash="dash", line_color="red", annotation_text="Threshold")
        
        fig.update_layout(
            title=f"7-Day Staffing Pressure Forecast: {selected_facility}",
            xaxis_title="Date",
            yaxis_title="Staffing Pressure",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("📥 Download Results")
        csv_staffing = predictions.to_csv(index=False)
        st.download_button(
            label="📄 Download Staffing Predictions (CSV)",
            data=csv_staffing,
            file_name=f"staffing_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key='staffing_download'
        )
    else:
        st.info("Upload data and run predictions to see staffing forecasts")

# ============================================================================
# TAB 2: PATIENT READMISSION RISK
# ============================================================================
with tab2:
    st.header("🔄 Patient Readmission Risk Prediction")
    st.markdown("Identify patients at high risk of 30-day readmission")
    
    # Sidebar for Tab 2
    with st.sidebar:
        st.subheader("📤 Readmission Data Input")
        
        readmission_file = st.file_uploader(
            "Upload patient visit CSV",
            type=['csv'],
            key='readmission_upload',
            help="CSV with patient demographics, visit details, and clinical factors"
        )
        
        st.divider()
        
        st.subheader("ℹ️ Model Info")
        st.markdown("""
        **Model:** LightGBM Classifier  
        **ROC-AUC:** 0.789  
        **Optimal Threshold:** 0.55  
        **Target:** 30-day readmission  
        **Class Balance:** 9.5% positive
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if readmission_file is not None:
            try:
                df_readmission = pd.read_csv(readmission_file)
                
                st.subheader("📄 Data Preview")
                st.dataframe(df_readmission.head(), use_container_width=True)
                st.info(f"✅ Loaded {len(df_readmission)} patients with {len(df_readmission.columns)} columns")
                
                # Check required columns
                required_columns = [
                    'patient_id', 'visit_id', 'arrival_datetime', 'discharge_datetime',
                    'triage_category', 'visit_type', 'age', 'sex',
                    'known_chronic_condition', 'num_chronic_diagnoses', 'num_procedures'
                ]
                
                missing_columns = [col for col in required_columns if col not in df_readmission.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                else:
                    if st.button("🚀 Generate Readmission Predictions", type="primary", key='readmission_predict'):
                        with st.spinner("Making predictions..."):
                            try:
                                predictor = ReadmissionPredictor()
                                predictor.load_model()
                                
                                predictions_readmit = predictor.predict(df_readmission)
                                
                                # Display results
                                st.subheader("📊 Prediction Results")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    total_patients = len(predictions_readmit)
                                    st.metric("Total Patients", f"{total_patients:,}")
                                
                                with col2:
                                    high_risk = (predictions_readmit['predicted_readmission'] == 1).sum()
                                    st.metric("High-Risk Patients", f"{high_risk:,}")
                                
                                with col3:
                                    high_risk_pct = (high_risk / total_patients * 100) if total_patients > 0 else 0
                                    st.metric("High-Risk %", f"{high_risk_pct:.1f}%")
                                
                                # Risk score distribution
                                st.subheader("📈 Risk Score Distribution")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Mean Risk Score", 
                                             f"{predictions_readmit['readmission_risk_score'].mean():.3f}")
                                    st.metric("Min Risk Score",
                                             f"{predictions_readmit['readmission_risk_score'].min():.3f}")
                                
                                with col2:
                                    st.metric("Median Risk Score",
                                             f"{predictions_readmit['readmission_risk_score'].median():.3f}")
                                    st.metric("Max Risk Score",
                                             f"{predictions_readmit['readmission_risk_score'].max():.3f}")
                                
                                # Show predictions table
                                st.subheader("👥 Patient Predictions")
                                
                                display_cols = ['patient_id', 'visit_id', 'age', 
                                              'triage_category', 'visit_type',
                                              'readmission_risk_score', 'predicted_readmission']
                                
                                if 'readmitted_30d' in predictions_readmit.columns:
                                    display_cols.append('readmitted_30d')
                                
                                predictions_display = predictions_readmit[display_cols].copy()
                                predictions_display['readmission_risk_score'] = predictions_display['readmission_risk_score'].round(4)
                                
                                st.dataframe(predictions_display, use_container_width=True, height=400)
                                
                                # Download button
                                st.subheader("💾 Download Results")
                                
                                csv_readmit = predictions_readmit.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv_readmit,
                                    file_name="readmission_predictions.csv",
                                    mime="text/csv",
                                    type="primary",
                                    key='readmission_download'
                                )
                                
                                # Feature importance
                                st.subheader("🔍 Top Predictors")
                                importance_df = predictor.get_feature_importance()
                                st.dataframe(importance_df.head(10), use_container_width=True)
                                
                            except FileNotFoundError:
                                st.error("❌ Model file not found. Please train the model first.")
                            except Exception as e:
                                st.error(f"❌ Prediction error: {str(e)}")
                                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        else:
            st.info("Upload a CSV file to begin readmission prediction")
    
    with col2:
        st.subheader("📋 Expected Columns")
        
        column_info = pd.DataFrame({
            'Column': [
                'patient_id', 'visit_id', 'arrival_datetime', 'discharge_datetime',
                'triage_category', 'visit_type', 'age', 'sex',
                'known_chronic_condition', 'num_chronic_diagnoses', 'num_procedures'
            ],
            'Type': [
                'Text', 'Text', 'Datetime', 'Datetime',
                'Categorical', 'Categorical', 'Integer', 'Binary',
                'Boolean', 'Integer', 'Integer'
            ]
        })
        
        st.dataframe(column_info, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("🎯 Risk Interpretation")
        st.markdown("""
        **Risk Score Ranges:**
        - **< 0.3:** Low risk
        - **0.3 - 0.6:** Medium risk
        - **0.6 - 0.8:** High risk
        - **> 0.8:** Critical risk
        
        **Prediction:** Binary (1 = likely readmission)
        **Threshold:** 0.55 (optimized for F1-score)
        """)

# ============================================================================
# TAB 3: FACILITY OVERLOAD DETECTION
# ============================================================================
with tab3:
    st.header("⚠️ Facility Overload Detection")
    st.markdown("Detect sustained overload as a proxy for burnout risk and declining care quality")
    
    # Sidebar for Tab 3
    with st.sidebar:
        st.subheader("📤 Facility Data Input")
        
        overload_file = st.file_uploader(
            "Upload facility data CSV",
            type=['csv'],
            key='overload_upload',
            help="CSV with: facility_id, day, beds_occupied, beds_available, total_staff, patient_load"
        )
        
        st.divider()
        
        st.subheader("⚙️ Detection Parameters")
        
        occupancy_threshold = st.slider(
            "High Occupancy Threshold",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            key='occupancy_threshold',
            help="Occupancy ratio above this indicates high occupancy"
        )
        
        staffing_threshold_overload = st.slider(
            "Low Staffing Threshold",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            key='staffing_threshold_overload',
            help="Staff-to-patient ratio below this indicates low staffing"
        )
        
        min_consecutive_days = st.slider(
            "Min Consecutive Days",
            min_value=3,
            max_value=14,
            value=7,
            step=1,
            key='min_consecutive',
            help="Minimum consecutive overload days for 'sustained' classification"
        )
    
    # Main content
    if overload_file is not None:
        try:
            with st.spinner("Loading data..."):
                df_overload = pd.read_csv(overload_file)
                
                required_cols = ['facility_id', 'day', 'beds_occupied', 'beds_available', 
                               'total_staff', 'patient_load']
                missing_cols = [col for col in required_cols if col not in df_overload.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                df_overload['day'] = pd.to_datetime(df_overload['day'])
            
            # Display data preview
            st.subheader("📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df_overload):,}")
            with col2:
                st.metric("Facilities", df_overload['facility_id'].nunique())
            with col3:
                date_range = f"{df_overload['day'].min().date()} to {df_overload['day'].max().date()}"
                st.metric("Date Range", date_range)
            
            with st.expander("View Data Sample", expanded=False):
                st.dataframe(df_overload.head(), use_container_width=True)
            
            # Process data
            if st.button("🚀 Analyze Overload Patterns", type="primary", key='overload_analyze'):
                with st.spinner("Analyzing facility data..."):
                    
                    progress_bar = st.progress(0)
                    
                    # Calculate metrics
                    progress_bar.progress(10)
                    df_overload['occupancy_ratio'] = df_overload['beds_occupied'] / df_overload['beds_available']
                    df_overload['staff_to_patient_ratio'] = df_overload['total_staff'] / df_overload['patient_load']
                    
                    df_overload = df_overload.sort_values(['facility_id', 'day'])
                    df_overload['occupancy_7d_avg'] = df_overload.groupby('facility_id')['occupancy_ratio'] \
                                                      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
                    
                    progress_bar.progress(30)
                    
                    # Calculate stress indicators
                    patient_load_threshold = df_overload['patient_load'].quantile(0.75)
                    
                    df_overload['high_occupancy_flag'] = (df_overload['occupancy_ratio'] > occupancy_threshold).astype(int)
                    df_overload['low_staffing_flag'] = (df_overload['staff_to_patient_ratio'] < staffing_threshold_overload).astype(int)
                    df_overload['high_patient_load_flag'] = (df_overload['patient_load'] > patient_load_threshold).astype(int)
                    df_overload['high_7d_avg_flag'] = (df_overload['occupancy_7d_avg'] > occupancy_threshold).astype(int)
                    
                    progress_bar.progress(50)
                    
                    # Composite score
                    indicators = ['high_occupancy_flag', 'low_staffing_flag', 
                                'high_patient_load_flag', 'high_7d_avg_flag']
                    df_overload['stress_score'] = df_overload[indicators].sum(axis=1)
                    df_overload['overload_day'] = (df_overload['stress_score'] >= 2).astype(int)
                    
                    progress_bar.progress(70)
                    
                    # Detect sustained periods
                    def find_sustained_periods(facility_data):
                        facility_data = facility_data.sort_values('day')
                        periods = []
                        current_streak = 0
                        streak_start = None
                        
                        for idx, row in facility_data.iterrows():
                            if row['overload_day'] == 1:
                                if current_streak == 0:
                                    streak_start = row['day']
                                current_streak += 1
                            else:
                                if current_streak >= min_consecutive_days:
                                    streak_end = facility_data.loc[idx-1, 'day'] if idx > 0 else streak_start
                                    periods.append({
                                        'start_date': streak_start,
                                        'end_date': streak_end,
                                        'duration_days': current_streak
                                    })
                                current_streak = 0
                                streak_start = None
                        
                        if current_streak >= min_consecutive_days:
                            streak_end = facility_data.iloc[-1]['day']
                            periods.append({
                                'start_date': streak_start,
                                'end_date': streak_end,
                                'duration_days': current_streak
                            })
                        
                        return periods
                    
                    df_overload['in_sustained_period'] = 0
                    all_periods = []
                    
                    for facility_id in df_overload['facility_id'].unique():
                        facility_data = df_overload[df_overload['facility_id'] == facility_id].copy()
                        sustained_periods = find_sustained_periods(facility_data)
                        all_periods.extend(sustained_periods)
                        
                        for period in sustained_periods:
                            mask = (df_overload['facility_id'] == facility_id) & \
                                   (df_overload['day'] >= period['start_date']) & \
                                   (df_overload['day'] <= period['end_date'])
                            df_overload.loc[mask, 'in_sustained_period'] = 1
                    
                    progress_bar.progress(90)
                    
                    # Calculate facility summaries
                    facility_summary = []
                    for facility_id in df_overload['facility_id'].unique():
                        facility_data = df_overload[df_overload['facility_id'] == facility_id]
                        
                        total_days = len(facility_data)
                        overload_days = facility_data['overload_day'].sum()
                        sustained_days = facility_data['in_sustained_period'].sum()
                        
                        overload_pct = (overload_days / total_days) * 100
                        sustained_pct = (sustained_days / total_days) * 100
                        avg_stress = facility_data['stress_score'].mean()
                        
                        if overload_pct > 60 and sustained_pct > 20:
                            risk = "CRITICAL"
                        elif overload_pct > 50 and sustained_pct > 10:
                            risk = "HIGH"
                        elif overload_pct > 40:
                            risk = "MODERATE"
                        else:
                            risk = "LOW"
                        
                        facility_summary.append({
                            'facility_id': facility_id,
                            'total_days': total_days,
                            'overload_days': overload_days,
                            'overload_percentage': round(overload_pct, 1),
                            'sustained_days': sustained_days,
                            'sustained_percentage': round(sustained_pct, 1),
                            'mean_stress_score': round(avg_stress, 2),
                            'burnout_risk': risk
                        })
                    
                    facility_summary_df = pd.DataFrame(facility_summary)
                    facility_summary_df = facility_summary_df.sort_values('overload_percentage', ascending=False)
                    
                    progress_bar.progress(100)
                    
                    # DISPLAY RESULTS
                    st.success("Analysis Complete!")
                    
                    # Key Metrics
                    st.subheader("📈 Key Metrics")
                    total_overload = df_overload['overload_day'].sum()
                    total_sustained = df_overload['in_sustained_period'].sum()
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Overload Days", f"{total_overload:,}", 
                                 f"{total_overload/len(df_overload)*100:.1f}%")
                    with m2:
                        st.metric("Sustained Overload Days", f"{total_sustained:,}", 
                                 f"{total_sustained/len(df_overload)*100:.1f}%")
                    with m3:
                        st.metric("Facilities at Risk", 
                                 f"{len([r for r in facility_summary if r['burnout_risk'] in ['CRITICAL', 'HIGH']])}")
                    with m4:
                        avg_stress = df_overload['stress_score'].mean()
                        st.metric("Average Stress Score", f"{avg_stress:.2f}")
                    
                    # Facility Ranking
                    st.subheader("🏆 Facility Overload Ranking")
                    st.dataframe(facility_summary_df, use_container_width=True)
                    
                    # Sustained Periods
                    st.subheader("📅 Sustained Overload Periods")
                    if all_periods:
                        periods_df = pd.DataFrame(all_periods)
                        periods_df['start_date'] = periods_df['start_date'].dt.date
                        periods_df['end_date'] = periods_df['end_date'].dt.date
                        periods_df = periods_df.sort_values('duration_days', ascending=False)
                        
                        st.dataframe(periods_df.head(10), use_container_width=True)
                        if len(periods_df) > 10:
                            st.caption(f"Showing top 10 of {len(periods_df)} sustained periods")
                    else:
                        st.info("No sustained overload periods detected")
                    
                    # Downloads
                    st.subheader("📥 Download Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_full = df_overload.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results",
                            data=csv_full,
                            file_name=f"overload_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key='overload_download_full'
                        )
                    
                    with col2:
                        csv_summary = facility_summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download Summary",
                            data=csv_summary,
                            file_name=f"overload_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key='overload_download_summary'
                        )
                    
                    with col3:
                        if all_periods:
                            csv_periods = periods_df.to_csv(index=False)
                            st.download_button(
                                label="Download Periods",
                                data=csv_periods,
                                file_name=f"sustained_periods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key='overload_download_periods'
                            )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and required columns.")
    
    else:
        st.info("👈 Upload a CSV file using the sidebar to begin analysis")
        
        with st.expander("📋 Expected CSV Format", expanded=True):
            st.markdown("""
            Your CSV should contain these columns:
            
            | Column | Description | Example |
            |--------|-------------|---------|
            | `facility_id` | Facility identifier | `HF_L4_004` |
            | `day` | Date (YYYY-MM-DD) | `2023-01-01` |
            | `beds_occupied` | Occupied beds | `9799.0` |
            | `beds_available` | Available beds | `4633` |
            | `total_staff` | Total staff count | `2255` |
            | `patient_load` | Total patient load | `49964.0` |
            """)
            
            example_data = pd.DataFrame({
                'facility_id': ['HF_L4_004', 'HF_L4_004'],
                'day': ['2023-01-01', '2023-01-02'],
                'beds_occupied': [9799.0, 7952.0],
                'beds_available': [4633, 1624],
                'total_staff': [2255, 1400],
                'patient_load': [49964.0, 35160.0]
            })
            st.dataframe(example_data, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 1rem;'>
    <p><strong>Workforce Optimization Dashboard v1.0</strong></p>
    <p>Predictive analytics for healthcare workforce planning | Staffing • Readmissions • Capacity</p>
</div>
""", unsafe_allow_html=True)
