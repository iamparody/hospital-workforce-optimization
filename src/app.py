"""
Streamlit Dashboard for Hospital Staffing Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import json

# Add src to path
sys.path.append('src')

# Set page config
st.set_page_config(
    page_title="Hospital Staffing Intelligence",
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
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Hospital Staffing Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Predict staffing shortages 7 days in advance for proactive resource allocation")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'features' not in st.session_state:
    st.session_state.features = None

# Sidebar
with st.sidebar:
    st.header("📤 Data Input")
    
    # Option 1: Upload CSV
    uploaded_file = st.file_uploader(
        "Upload daily metrics CSV",
        type=['csv'],
        help="Upload CSV with columns: facility_id, day, total_staff, nurses_on_duty, doctors_on_duty, beds_occupied, beds_available, patient_visits, avg_congestion, weighted_demand, staff_to_patient_ratio, beds_occupancy_ratio"
    )
    
    # Option 2: Sample data
    use_sample = st.checkbox("Use sample data", value=False)
    
    st.divider()
    
    # Prediction controls
    st.header("⚙️ Prediction Settings")
    threshold = st.slider(
        "Risk threshold (%)",
        min_value=30,
        max_value=70,
        value=50,
        help="Probability above which facility is considered 'at risk'"
    )
    
    days_ahead = st.slider(
        "Forecast horizon (days)",
        min_value=3,
        max_value=14,
        value=7,
        help="Number of days to predict ahead"
    )
    
    st.divider()
    
    # Run prediction button
    predict_btn = st.button(
        "🚀 Run Predictions",
        type="primary",
        use_container_width=True
    )
    
    st.divider()
    
    # Info
    st.info("""
    **How it works:**
    1. Upload daily hospital metrics
    2. Click 'Run Predictions'
    3. View 7-day staffing forecasts
    4. Download results
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "📈 Trends", "📁 Data"])

with tab1:
    # Dashboard overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Facilities Monitored",
            "5" if st.session_state.predictions is not None else "0",
            "Active" if st.session_state.predictions is not None else "Upload data"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.predictions is not None:
            at_risk = st.session_state.predictions[
                st.session_state.predictions['predicted_understaffed']
            ]['facility_id'].nunique()
            st.metric("At Risk Facilities", at_risk, "")
        else:
            st.metric("At Risk Facilities", "0", "Upload data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.predictions is not None:
            total_days = len(st.session_state.predictions)
            risk_days = st.session_state.predictions['predicted_understaffed'].sum()
            st.metric("Risk Days Next Week", f"{risk_days}/{total_days}", "")
        else:
            st.metric("Risk Days Next Week", "0/0", "Upload data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk heatmap
    st.subheader("📅 7-Day Risk Forecast")
    
    if st.session_state.predictions is not None:
        # Create heatmap data
        heatmap_data = st.session_state.predictions.pivot_table(
            index='facility_id',
            columns='ds',
            values='understaffing_probability',
            aggfunc='mean'
        )
        
        # Plot heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            labels=dict(color="Risk %"),
            title="Understaffing Risk by Facility and Day"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data and run predictions to see the risk forecast")
    
    # Facility risk summary
    st.subheader("🏥 Facility Risk Summary")
    
    if st.session_state.predictions is not None:
        # Calculate summary metrics
        summary = st.session_state.predictions.groupby('facility_id').agg({
            'understaffing_probability': 'max',
            'predicted_understaffed': 'any',
            'yhat': 'mean'
        }).reset_index()
        
        summary.columns = ['Facility', 'Max Risk %', 'At Risk', 'Avg Pressure']
        summary['Max Risk %'] = (summary['Max Risk %'] * 100).round(1)
        summary['Avg Pressure'] = summary['Avg Pressure'].round(1)
        
        # Add risk level
        def get_risk_level(risk_pct):
            if risk_pct >= threshold:
                return "🟥 High"
            elif risk_pct >= threshold - 15:
                return "🟨 Medium"
            else:
                return "🟩 Low"
        
        summary['Risk Level'] = summary['Max Risk %'].apply(get_risk_level)
        
        # Display table
        st.dataframe(
            summary,
            column_config={
                "Facility": st.column_config.TextColumn("Facility"),
                "Max Risk %": st.column_config.ProgressColumn(
                    "Max Risk %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "At Risk": st.column_config.CheckboxColumn("At Risk"),
                "Avg Pressure": st.column_config.NumberColumn("Avg Pressure"),
                "Risk Level": st.column_config.TextColumn("Risk Level")
            },
            use_container_width=True
        )
    else:
        st.info("No predictions available. Upload data and run predictions.")

with tab2:
    # Detailed predictions
    st.subheader("🔍 Detailed Predictions")
    
    if st.session_state.predictions is not None:
        # Facility selector
        facilities = st.session_state.predictions['facility_id'].unique()
        selected_facility = st.selectbox("Select Facility", facilities)
        
        # Filter predictions
        facility_preds = st.session_state.predictions[
            st.session_state.predictions['facility_id'] == selected_facility
        ].copy()
        
        # Convert probabilities to percentages
        facility_preds['Risk %'] = (facility_preds['understaffing_probability'] * 100).round(1)
        
        # Plot forecast
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=facility_preds['ds'],
            y=facility_preds['yhat'],
            mode='lines+markers',
            name='Predicted Pressure',
            line=dict(color='#3B82F6', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([facility_preds['ds'], facility_preds['ds'][::-1]]),
            y=pd.concat([facility_preds['yhat_upper'], facility_preds['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        # Add threshold line
        threshold_value = 97.03  # From your config
        fig.add_hline(
            y=threshold_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Understaffing Threshold"
        )
        
        fig.update_layout(
            title=f"7-Day Staffing Pressure Forecast: {selected_facility}",
            xaxis_title="Date",
            yaxis_title="Staffing Pressure",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction table
        st.subheader("📋 Day-by-Day Predictions")
        
        display_cols = ['ds', 'yhat', 'Risk %', 'predicted_understaffed']
        display_df = facility_preds[display_cols].copy()
        display_df.columns = ['Date', 'Pressure', 'Risk %', 'Understaffed']
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Pressure'] = display_df['Pressure'].round(1)
        display_df['Understaffed'] = display_df['Understaffed'].map({True: 'Yes', False: 'No'})
        
        st.dataframe(display_df, use_container_width=True)
        
        # Risk days summary
        risk_days = facility_preds['predicted_understaffed'].sum()
        max_risk = facility_preds['Risk %'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Understaffed Days", f"{risk_days} of 7")
        with col2:
            st.metric("Maximum Risk", f"{max_risk}%")
        
    else:
        st.info("Run predictions to see detailed forecasts")

with tab3:
    # Historical trends
    st.subheader("📈 Historical Trends")
    
    if st.session_state.uploaded_data is not None:
        # Facility selector
        facilities = st.session_state.uploaded_data['facility_id'].unique()
        selected_facility_trend = st.selectbox("Select Facility for Trends", facilities, key='trend_facility')
        
        # Filter data
        facility_data = st.session_state.uploaded_data[
            st.session_state.uploaded_data['facility_id'] == selected_facility_trend
        ].copy()
        
        # Calculate staffing pressure if not already
        if 'staffing_pressure' not in facility_data.columns:
            facility_data['staffing_pressure'] = (
                facility_data['staff_to_patient_ratio'] * facility_data['weighted_demand']
            )
        
        # Plot historical pressure
        fig = px.line(
            facility_data,
            x='day',
            y='staffing_pressure',
            title=f"Historical Staffing Pressure: {selected_facility_trend}",
            markers=True
        )
        
        # Add threshold line
        fig.add_hline(
            y=97.03,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold"
        )
        
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Staffing Pressure")
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics over time
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = px.line(facility_data, x='day', y='staff_to_patient_ratio',
                          title="Staff-to-Patient Ratio")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(facility_data, x='day', y='weighted_demand',
                          title="Weighted Demand")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            fig3 = px.line(facility_data, x='day', y='beds_occupancy_ratio',
                          title="Bed Occupancy")
            st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.info("Upload data to see historical trends")

with tab4:
    # Data management
    st.subheader("📁 Data Management")
    
    if uploaded_file is not None:
        # Load and display uploaded data
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        
        st.success(f"✅ Data loaded: {len(df)} rows, {df['facility_id'].nunique()} facilities")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Facilities", df['facility_id'].nunique())
        with col3:
            date_range = f"{df['day'].min()} to {df['day'].max()}"
            st.metric("Date Range", date_range)
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notna().sum(),
            'Null %': (df.isna().sum() / len(df) * 100).round(1)
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Run feature engineering
        if st.button("🛠️ Engineer Features", type="secondary"):
            with st.spinner("Engineering features..."):
                try:
                    from feature_engineering import create_features
                    features_df = create_features(df)
                    st.session_state.features = features_df
                    st.success(f"✅ Created {len(features_df.columns)} features")
                    
                    # Show feature sample
                    st.subheader("Engineered Features Sample")
                    st.dataframe(features_df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        st.info("Upload a CSV file to view and manage data")
    
    # Download predictions
    if st.session_state.predictions is not None:
        st.divider()
        st.subheader("📥 Download Results")
        
        # Convert predictions to CSV
        predictions_csv = st.session_state.predictions.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Predictions (CSV)",
                data=predictions_csv,
                file_name=f"staffing_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create summary
            summary = {
                'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'total_facilities': st.session_state.predictions['facility_id'].nunique(),
                'at_risk_facilities': st.session_state.predictions[
                    st.session_state.predictions['predicted_understaffed']
                ]['facility_id'].nunique(),
                'total_risk_days': st.session_state.predictions['predicted_understaffed'].sum()
            }
            
            st.download_button(
                label="📊 Download Summary (JSON)",
                data=json.dumps(summary, indent=2),
                file_name=f"staffing_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

# Run predictions when button is clicked
if predict_btn and st.session_state.uploaded_data is not None:
    with st.spinner("Running predictions..."):
        try:
            # Import prediction functions
            from feature_engineering import create_features
            from model_predict import predict_next_7_days, generate_executive_summary
            
            # Engineer features if not already done
            if st.session_state.features is None:
                features_df = create_features(st.session_state.uploaded_data)
                st.session_state.features = features_df
            
            # Make predictions
            predictions = predict_next_7_days(st.session_state.features, feature_engineered=True)
            
            # Apply custom threshold
            predictions['predicted_understaffed'] = predictions['understaffing_probability'] > (threshold / 100)
            
            # Store in session state
            st.session_state.predictions = predictions
            
            # Show success message
            st.success(f"✅ Predictions generated for {predictions['facility_id'].nunique()} facilities")
            
            # Refresh the page to show new predictions
            st.rerun()
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Make sure models are in '1_data_generation/saved_models/' directory")

# Footer
st.divider()
st.caption("""
**Hospital Staffing Intelligence System** | 
Predicts staffing shortages 7 days in advance using Prophet time-series models | 
Refresh data daily for updated predictions
""")