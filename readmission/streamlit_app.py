"""
streamlit_app.py
Simple Streamlit UI for readmission prediction from CSV upload.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from readmission_predictor import ReadmissionPredictor
import io

# Page configuration
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Patient 30-Day Readmission Predictor")
st.markdown("""
This tool predicts which patients are most likely to be readmitted within 30 days.
Upload a CSV file with patient data to get predictions.
""")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Prepare your CSV file** with the following columns:
       - `patient_id`: Patient identifier
       - `visit_id`: Visit identifier
       - `arrival_datetime`: Arrival timestamp
       - `discharge_datetime`: Discharge timestamp
       - `triage_category`: Triage level (RED/YELLOW/GREEN/ORANGE)
       - `visit_type`: Visit type (emergency/outpatient/inpatient)
       - `next_visit`: Next appointment date (can be empty)
       - `age`: Patient age
       - `sex`: Patient sex
       - `known_chronic_condition`: Boolean
       - `num_chronic_diagnoses`: Number of chronic diagnoses
       - `num_procedures`: Number of procedures
       - `readmitted_30d`: Actual readmission (for validation, optional)
       - `days_until_next_visit`: Days until next visit
    
    2. **Upload your CSV file** below
    3. **Download predictions** as CSV
    """)
    
    st.divider()
    
    st.header("⚙️ Model Info")
    st.markdown("""
    **Model:** LightGBM Classifier  
    **ROC-AUC:** 0.789  
    **Optimal Threshold:** 0.55  
    **Target:** 30-day readmission  
    **Class Balance:** 9.5% positive
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload Patient Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload patient data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Show preview
            st.subheader("📄 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.info(f"✅ Loaded {len(df)} patients with {len(df.columns)} columns")
            
            # Check required columns
            required_columns = [
                'patient_id', 'visit_id', 'arrival_datetime', 'discharge_datetime',
                'triage_category', 'visit_type', 'age', 'sex',
                'known_chronic_condition', 'num_chronic_diagnoses', 'num_procedures'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
            else:
                # Make predictions
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Initialize predictor
                        predictor = ReadmissionPredictor()
                        
                        try:
                            # Load pre-trained model
                            predictor.load_model()
                            
                            # Make predictions
                            predictions = predictor.predict(df)
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_patients = len(predictions)
                                st.metric("Total Patients", f"{total_patients:,}")
                            
                            with col2:
                                high_risk = (predictions['predicted_readmission'] == 1).sum()
                                st.metric("High-Risk Patients", f"{high_risk:,}")
                            
                            with col3:
                                high_risk_pct = (high_risk / total_patients * 100) if total_patients > 0 else 0
                                st.metric("High-Risk %", f"{high_risk_pct:.1f}%")
                            
                            # Risk score distribution
                            st.subheader("📈 Risk Score Distribution")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Mean Risk Score", 
                                         f"{predictions['readmission_risk_score'].mean():.3f}")
                                st.metric("Min Risk Score",
                                         f"{predictions['readmission_risk_score'].min():.3f}")
                            
                            with col2:
                                st.metric("Median Risk Score",
                                         f"{predictions['readmission_risk_score'].median():.3f}")
                                st.metric("Max Risk Score",
                                         f"{predictions['readmission_risk_score'].max():.3f}")
                            
                            # Show predictions table
                            st.subheader("👥 Patient Predictions")
                            
                            display_cols = ['patient_id', 'visit_id', 'age', 
                                          'triage_category', 'visit_type',
                                          'readmission_risk_score', 'predicted_readmission']
                            
                            if 'readmitted_30d' in predictions.columns:
                                display_cols.append('readmitted_30d')
                            
                            predictions_display = predictions[display_cols].copy()
                            predictions_display['readmission_risk_score'] = predictions_display['readmission_risk_score'].round(4)
                            
                            # Color code by risk
                            def highlight_risk(row):
                                if row['predicted_readmission'] == 1:
                                    return ['background-color: #ffcccc'] * len(row)
                                elif row['readmission_risk_score'] > 0.7:
                                    return ['background-color: #fff3cd'] * len(row)
                                return [''] * len(row)
                            
                            st.dataframe(
                                predictions_display.style.apply(highlight_risk, axis=1),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download button
                            st.subheader("💾 Download Results")
                            
                            csv = predictions.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="readmission_predictions.csv",
                                mime="text/csv",
                                type="primary"
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

with col2:
    st.header("📋 Expected Columns")
    
    column_info = pd.DataFrame({
        'Column': [
            'patient_id', 'visit_id', 'arrival_datetime', 'discharge_datetime',
            'triage_category', 'visit_type', 'next_visit', 'age', 'sex',
            'known_chronic_condition', 'num_chronic_diagnoses', 'num_procedures',
            'readmitted_30d', 'days_until_next_visit'
        ],
        'Type': [
            'Text', 'Text', 'Datetime', 'Datetime',
            'Categorical', 'Categorical', 'Datetime (optional)', 'Integer', 'Binary',
            'Boolean', 'Integer', 'Integer',
            'Binary (optional)', 'Float (optional)'
        ],
        'Description': [
            'Unique patient identifier',
            'Unique visit identifier',
            'Patient arrival time',
            'Patient discharge time',
            'Triage severity level',
            'Type of visit',
            'Next scheduled visit',
            'Patient age in years',
            'Patient sex (0/1)',
            'Has chronic condition',
            'Number of chronic diagnoses',
            'Number of procedures performed',
            'Actual readmission status',
            'Days until next visit'
        ]
    })
    
    st.dataframe(column_info, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.header("🎯 Risk Interpretation")
    st.markdown("""
    **Risk Score Ranges:**
    - **< 0.3:** Low risk
    - **0.3 - 0.6:** Medium risk
    - **0.6 - 0.8:** High risk
    - **> 0.8:** Critical risk
    
    **Prediction:** Binary (1 = likely readmission)
    **Threshold:** 0.55 (optimized for F1-score)
    """)

# Footer
st.divider()
st.caption("""
**Note:** This is a predictive model for decision support. 
Clinical judgment should always be used in patient care decisions.
Model performance: ROC-AUC = 0.789, PR-AUC = 0.237
""")