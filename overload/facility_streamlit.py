#!/usr/bin/env python3
"""
Streamlit Dashboard for Facility Overload Detection
Simple upload interface with results display.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile

# Page config
st.set_page_config(
    page_title="Facility Overload Detector",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Healthcare Facility Overload Detection")
st.markdown("""
Detect sustained overload in healthcare facilities as a proxy for burnout risk and declining care quality.
Upload facility data CSV to analyze overload patterns.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Upload & Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Facility Data CSV",
        type=['csv'],
        help="CSV should contain: facility_id, day, beds_occupied, beds_available, total_staff, patient_load"
    )
    
    st.markdown("---")
    st.header("Detection Parameters")
    
    # Threshold configuration
    occupancy_threshold = st.slider(
        "High Occupancy Threshold",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Occupancy ratio above this value indicates high occupancy"
    )
    
    staffing_threshold = st.slider(
        "Low Staffing Threshold",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Staff-to-patient ratio below this value indicates low staffing"
    )
    
    min_consecutive_days = st.slider(
        "Minimum Consecutive Days for Sustained Overload",
        min_value=3,
        max_value=14,
        value=7,
        step=1,
        help="Minimum number of consecutive overload days to count as 'sustained'"
    )
    
    st.markdown("---")
    st.caption("v1.0 | Overload Detection System")

# Main content area
if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['facility_id', 'day', 'beds_occupied', 'beds_available', 
                           'total_staff', 'patient_load']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Convert date column
            df['day'] = pd.to_datetime(df['day'])
            
        # Display data preview
        st.subheader("📊 Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Facilities", df['facility_id'].nunique())
        with col3:
            date_range = f"{df['day'].min().date()} to {df['day'].max().date()}"
            st.metric("Date Range", date_range)
        
        # Show data sample
        with st.expander("View Data Sample", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
        
        # Process data button
        if st.button("🚀 Analyze Overload Patterns", type="primary"):
            with st.spinner("Analyzing facility data..."):
                
                # Progress bar
                progress_bar = st.progress(0)
                
                # Step 1: Calculate basic metrics
                progress_bar.progress(10)
                df['occupancy_ratio'] = df['beds_occupied'] / df['beds_available']
                df['staff_to_patient_ratio'] = df['total_staff'] / df['patient_load']
                
                # Calculate rolling averages
                df = df.sort_values(['facility_id', 'day'])
                df['occupancy_7d_avg'] = df.groupby('facility_id')['occupancy_ratio'] \
                                          .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
                
                progress_bar.progress(30)
                
                # Step 2: Calculate stress indicators
                patient_load_threshold = df['patient_load'].quantile(0.75)
                
                df['high_occupancy_flag'] = (df['occupancy_ratio'] > occupancy_threshold).astype(int)
                df['low_staffing_flag'] = (df['staff_to_patient_ratio'] < staffing_threshold).astype(int)
                df['high_patient_load_flag'] = (df['patient_load'] > patient_load_threshold).astype(int)
                df['high_7d_avg_flag'] = (df['occupancy_7d_avg'] > occupancy_threshold).astype(int)
                
                progress_bar.progress(50)
                
                # Step 3: Composite overload score
                indicators = ['high_occupancy_flag', 'low_staffing_flag', 
                            'high_patient_load_flag', 'high_7d_avg_flag']
                df['stress_score'] = df[indicators].sum(axis=1)
                df['overload_day'] = (df['stress_score'] >= 2).astype(int)
                
                progress_bar.progress(70)
                
                # Step 4: Detect sustained periods
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
                    
                    # Check final streak
                    if current_streak >= min_consecutive_days:
                        streak_end = facility_data.iloc[-1]['day']
                        periods.append({
                            'start_date': streak_start,
                            'end_date': streak_end,
                            'duration_days': current_streak
                        })
                    
                    return periods
                
                # Add sustained period flag
                df['in_sustained_period'] = 0
                all_periods = []
                
                for facility_id in df['facility_id'].unique():
                    facility_data = df[df['facility_id'] == facility_id].copy()
                    sustained_periods = find_sustained_periods(facility_data)
                    all_periods.extend(sustained_periods)
                    
                    # Mark days in sustained periods
                    for period in sustained_periods:
                        mask = (df['facility_id'] == facility_id) & \
                               (df['day'] >= period['start_date']) & \
                               (df['day'] <= period['end_date'])
                        df.loc[mask, 'in_sustained_period'] = 1
                
                progress_bar.progress(90)
                
                # Step 5: Calculate facility summaries
                facility_summary = []
                for facility_id in df['facility_id'].unique():
                    facility_data = df[df['facility_id'] == facility_id]
                    
                    total_days = len(facility_data)
                    overload_days = facility_data['overload_day'].sum()
                    sustained_days = facility_data['in_sustained_period'].sum()
                    
                    # Calculate risk level
                    overload_pct = (overload_days / total_days) * 100
                    sustained_pct = (sustained_days / total_days) * 100
                    avg_stress = facility_data['stress_score'].mean()
                    
                    # Simple risk assessment
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
                
                # RESULTS DISPLAY
                st.success("Analysis Complete!")
                
                # Key Metrics
                st.subheader("📈 Key Metrics")
                total_overload = df['overload_day'].sum()
                total_sustained = df['in_sustained_period'].sum()
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Overload Days", f"{total_overload:,}", 
                             f"{total_overload/len(df)*100:.1f}%")
                with m2:
                    st.metric("Sustained Overload Days", f"{total_sustained:,}", 
                             f"{total_sustained/len(df)*100:.1f}%")
                with m3:
                    st.metric("Facilities at Risk", 
                             f"{len([r for r in facility_summary if r['burnout_risk'] in ['CRITICAL', 'HIGH']])}")
                with m4:
                    avg_stress = df['stress_score'].mean()
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
                
                # Data Download Section
                st.subheader("📥 Download Results")
                
                # Create download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Full results CSV
                    csv_full = df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv_full,
                        file_name=f"overload_results_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Summary CSV
                    csv_summary = facility_summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Facility Summary (CSV)",
                        data=csv_summary,
                        file_name=f"overload_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Sustained periods CSV
                    if all_periods:
                        csv_periods = periods_df.to_csv(index=False)
                        st.download_button(
                            label="Download Sustained Periods (CSV)",
                            data=csv_periods,
                            file_name=f"sustained_periods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Analysis Report
                with st.expander("📋 Analysis Report", expanded=False):
                    st.markdown(f"""
                    ### Overload Detection Analysis Report
                    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    **File:** {uploaded_file.name}
                    
                    #### Executive Summary
                    - **Facilities Analyzed:** {df['facility_id'].nunique()}
                    - **Total Days Analyzed:** {len(df):,}
                    - **Overall Overload Rate:** {total_overload/len(df)*100:.1f}%
                    - **Sustained Overload Rate:** {total_sustained/len(df)*100:.1f}%
                    
                    #### Risk Distribution
                    - **CRITICAL/HIGH Risk Facilities:** {len([r for r in facility_summary if r['burnout_risk'] in ['CRITICAL', 'HIGH']])}
                    - **MODERATE Risk Facilities:** {len([r for r in facility_summary if r['burnout_risk'] == 'MODERATE'])}
                    - **LOW Risk Facilities:** {len([r for r in facility_summary if r['burnout_risk'] == 'LOW'])}
                    
                    #### Detection Parameters Used
                    - High Occupancy Threshold: {occupancy_threshold}
                    - Low Staffing Threshold: {staffing_threshold}
                    - Minimum Consecutive Days: {min_consecutive_days}
                    """)
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and required columns.")

else:
    # Display instructions when no file uploaded
    st.info("👈 Please upload a CSV file using the sidebar to begin analysis.")
    
    # Example data format
    with st.expander("📋 Expected CSV Format", expanded=True):
        st.markdown("""
        Your CSV should contain these columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | `facility_id` | Facility identifier | `HF_L4_004` |
        | `day` | Date (YYYY-MM-DD) | `2023-01-01` |
        | `beds_occupied` | Number of occupied beds | `9799.0` |
        | `beds_available` | Number of available beds | `4633` |
        | `total_staff` | Total staff count | `2255` |
        | `patient_load` | Total patient load | `49964.0` |
        
        **Optional columns** (will be calculated if not present):
        - `occupancy_ratio`
        - `staff_to_patient_ratio`
        - `occupancy_7d_avg`
        - `occupancy_14d_avg`
        """)
        
        # Example data
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
st.markdown("---")
st.caption("Healthcare Facility Overload Detection System v1.0 | For operational risk assessment and burnout prevention")