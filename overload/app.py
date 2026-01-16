# app.py - Streamlit app for 30-day readmission prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # just for type hint
from imblearn.ensemble import BalancedRandomForestClassifier

# ─── CONFIGURATION ───────────────────────────────────────────────────────
MODEL_PATH = "balanced_random_forest_readmission.pkl"
FEATURES = [
    'age', 'sex_num', 'chronic_condition_num', 'num_procedures',
    'los_days', 'num_prior_visits', 'prev_los', 'prev_readmitted'
]

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────
def preprocess_for_prediction(raw_df):
    """Full preprocessing pipeline for raw procedure-level data"""
    df = raw_df.copy()
    
    # Convert datetime
    df['procedure_datetime'] = pd.to_datetime(df['procedure_datetime'], errors='coerce')
    
    # Create visit-level data
    visits = df.groupby('visit_id').agg({
        'patient_id': 'first',
        'procedure_datetime': ['min', 'max'],
        'procedure_code': 'count',
        'age': 'first',
        'sex': 'first',
        'known_chronic_condition': 'first'
    }).reset_index()
    
    visits.columns = [
        'visit_id', 'patient_id',
        'admission_datetime', 'discharge_datetime_raw',
        'num_procedures', 'age', 'sex', 'chronic_condition'
    ]
    
    # Simulate LOS (same as training)
    np.random.seed(42)
    los_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 30]
    los_probs = np.array([0.10, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.025, 0.015, 0.01, 0.005])
    los_probs = los_probs / los_probs.sum()
    
    visits['los_days'] = np.random.choice(los_values, size=len(visits), p=los_probs)
    
    # Small boost for multiple procedures
    multi_mask = visits['num_procedures'] > 3
    if multi_mask.any():
        visits.loc[multi_mask, 'los_days'] = np.random.choice(
            [5, 6, 7, 8, 9, 10, 14, 21],
            size=sum(multi_mask),
            p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
        )
    
    # Final dates
    visits['discharge_datetime'] = visits['admission_datetime'] + pd.to_timedelta(visits['los_days'], unit='D')
    visits['admission_date'] = visits['admission_datetime'].dt.date
    visits['discharge_date'] = visits['discharge_datetime'].dt.date
    
    # Sort by patient and admission time
    visits = visits.sort_values(['patient_id', 'admission_datetime']).reset_index(drop=True)
    
    # Add required engineered features
    # Calculate prior visits per patient
    visits['num_prior_visits'] = visits.groupby('patient_id').cumcount()
    
    # Calculate previous LOS (for patients with prior visits)
    visits['prev_los'] = visits.groupby('patient_id')['los_days'].shift(1).fillna(0)
    
    # Previous readmission flag (initialize as 0, will be calculated if we had labels)
    visits['prev_readmitted'] = 0
    
    # Numeric encoding
    visits['sex_num'] = visits['sex'].map({'M': 0, 'F': 1}).fillna(0.5)
    visits['chronic_condition_num'] = visits['chronic_condition'].astype(int)
    
    # Select only model features
    return visits[FEATURES]

def load_model():
    """Load the pre-trained Balanced Random Forest model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ─── STREAMLIT APP ───────────────────────────────────────────────────────

st.set_page_config(page_title="30-Day Readmission Risk Predictor", layout="wide")

st.title("30-Day Hospital Readmission Risk Predictor")
st.markdown("""
Upload a CSV file containing patient visit data.  
The model will predict the probability of readmission within 30 days.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Processing your data..."):
        try:
            # Read uploaded data
            df = pd.read_csv(uploaded_file)
            
            # Check if this is raw data or already processed
            # Raw data has: visit_id, patient_id, procedure_datetime, procedure_code, etc.
            # Processed data has: age, sex_num, chronic_condition_num, etc.
            
            if 'procedure_datetime' in df.columns:
                # This is raw procedure-level data - need to process it
                st.info("Detected raw procedure data. Processing to create features...")
                X_pred = preprocess_for_prediction(df)
                
            elif all(feat in df.columns for feat in FEATURES):
                # This is already processed with engineered features
                st.info("Detected already processed data with engineered features.")
                X_pred = df[FEATURES]
                
            else:
                # Missing columns - show what's available vs what's needed
                st.error("Data format not recognized.")
                st.write("**Available columns in your file:**")
                st.write(list(df.columns))
                st.write("**Required for raw data:** `visit_id`, `patient_id`, `procedure_datetime`, `procedure_code`, `age`, `sex`, `known_chronic_condition`")
                st.write("**Required for processed data:**", FEATURES)
                st.stop()
            
            # Load model
            model = load_model()
            
            # Predict
            probabilities = model.predict_proba(X_pred)[:, 1]  # probability of class 1 (readmission)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'readmission_probability': probabilities,
                'readmission_risk': (probabilities >= 0.3).astype(int)  # Lower threshold for imbalance
            })
            
            # Add patient ID and visit ID if available
            if 'patient_id' in df.columns and len(df) == len(results_df):
                results_df['patient_id'] = df['patient_id'].values
            if 'visit_id' in df.columns and len(df) == len(results_df):
                results_df['visit_id'] = df['visit_id'].values
            
            st.success(f"Prediction complete! Processed {len(results_df)} visits.")
            
            # Show results
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Display statistics
            st.subheader("Risk Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_risk = (probabilities >= 0.5).sum()
                st.metric("High Risk (≥50%)", f"{high_risk}")
            
            with col2:
                medium_risk = ((probabilities >= 0.3) & (probabilities < 0.5)).sum()
                st.metric("Medium Risk (30-50%)", f"{medium_risk}")
            
            with col3:
                avg_risk = probabilities.mean()
                st.metric("Average Risk", f"{avg_risk:.1%}")
            
            with col4:
                max_risk = probabilities.max()
                st.metric("Highest Risk", f"{max_risk:.1%}")
            
            # Risk distribution chart
            st.subheader("Risk Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            ax.hist(probabilities, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=0.3, color='orange', linestyle='--', label='30% Risk Threshold')
            ax.axvline(x=0.5, color='red', linestyle='--', label='50% Risk Threshold')
            
            ax.set_xlabel("Readmission Probability")
            ax.set_ylabel("Number of Patients")
            ax.set_title("Distribution of Predicted Readmission Probabilities")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Download results
            st.subheader("Download Predictions")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv,
                file_name="readmission_predictions.csv",
                mime="text/csv",
                help="Click to download the predictions"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            # Show more detailed error for debugging
            with st.expander("Show error details"):
                import traceback
                st.code(traceback.format_exc())

else:
    # Show data format instructions when no file is uploaded
    with st.expander("📋 Data Format Instructions", expanded=True):
        st.write("""
        ### Option 1: Upload Raw Procedure Data (Recommended)
        
        Upload the **raw procedure-level data** with these columns:
        - `visit_id`: Unique identifier for the hospital visit
        - `patient_id`: Unique identifier for the patient
        - `procedure_datetime`: Date and time of the procedure (format: YYYY-MM-DD HH:MM)
        - `procedure_code`: Code for the procedure (any format)
        - `age`: Patient's age (numeric)
        - `sex`: Patient's gender (M/F)
        - `known_chronic_condition`: 1 if patient has chronic condition, 0 otherwise
        
        **Example:**
        """)
        
        sample_data = pd.DataFrame({
            'visit_id': ['V1001', 'V1001', 'V1002'],
            'patient_id': ['P001', 'P001', 'P002'],
            'procedure_datetime': ['2023-01-15 09:30', '2023-01-15 11:00', '2023-01-16 10:00'],
            'procedure_code': ['CPT-001', 'CPT-002', 'CPT-001'],
            'age': [65, 65, 72],
            'sex': ['M', 'M', 'F'],
            'known_chronic_condition': [1, 1, 0]
        })
        st.dataframe(sample_data)
        
        st.write("""
        ### Option 2: Upload Already-Processed Features
        
        If you have already calculated the engineered features, upload a CSV with these columns:
        - `age`: Patient's age
        - `sex_num`: Sex encoded as numeric (0=M, 1=F)
        - `chronic_condition_num`: 1 if chronic condition, 0 otherwise
        - `num_procedures`: Number of procedures during the visit
        - `los_days`: Length of stay in days
        - `num_prior_visits`: Number of previous visits for this patient
        - `prev_los`: Length of stay of previous visit (0 if first visit)
        - `prev_readmitted`: Whether patient was readmitted previously (0/1)
        """)
        
        # Download sample template
        st.write("**Download a sample template:**")
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            "Download Sample Template",
            data=csv_sample,
            file_name="sample_readmission_data.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Model: Balanced Random Forest | Trained on synthetic data | For demonstration only")
