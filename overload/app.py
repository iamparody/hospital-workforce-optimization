# app.py - Enhanced UI for 30-Day Readmission Risk Predictor

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier

# ─── CONFIG & STYLING ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look & feel
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stFileUploader label {font-size: 1.1rem; font-weight: bold;}
    .stMetric {background-color: #e8f4f8; border-radius: 10px; padding: 10px;}
    .risk-high {color: #d32f2f; font-weight: bold;}
    .risk-medium {color: #f57c00; font-weight: bold;}
    .risk-low {color: #388e3c; font-weight: bold;}
    .sidebar .sidebar-content {background-color: #ffffff; border-right: 1px solid #ddd;}
    </style>
""", unsafe_allow_html=True)

# ─── CONFIGURATION ───────────────────────────────────────────────────────
MODEL_PATH = "balanced_random_forest_readmission.pkl"
FEATURES = [
    'age', 'sex_num', 'chronic_condition_num', 'num_procedures',
    'los_days', 'num_prior_visits', 'prev_los', 'prev_readmitted'
]

# ─── HELPER FUNCTIONS (unchanged logic) ──────────────────────────────────
def preprocess_for_prediction(raw_df):
    df = raw_df.copy()
    df['procedure_datetime'] = pd.to_datetime(df['procedure_datetime'], errors='coerce')
    
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
    
    np.random.seed(42)
    los_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 30]
    los_probs = np.array([0.10, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.025, 0.015, 0.01, 0.005])
    los_probs /= los_probs.sum()
    
    visits['los_days'] = np.random.choice(los_values, size=len(visits), p=los_probs)
    
    multi_mask = visits['num_procedures'] > 3
    if multi_mask.any():
        visits.loc[multi_mask, 'los_days'] = np.random.choice(
            [5, 6, 7, 8, 9, 10, 14, 21],
            size=sum(multi_mask),
            p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
        )
    
    visits['discharge_datetime'] = visits['admission_datetime'] + pd.to_timedelta(visits['los_days'], unit='D')
    visits['admission_date'] = visits['admission_datetime'].dt.date
    visits['discharge_date'] = visits['discharge_datetime'].dt.date
    
    visits = visits.sort_values(['patient_id', 'admission_datetime']).reset_index(drop=True)
    
    visits['num_prior_visits'] = visits.groupby('patient_id').cumcount()
    visits['prev_readmitted'] = 0
    visits['prev_los'] = 0
    
    visits['sex_num'] = visits['sex'].map({'M': 0, 'F': 1}).fillna(0.5)
    visits['chronic_condition_num'] = visits['chronic_condition'].astype(int)
    
    return visits[FEATURES]

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
    st.title("Readmission Predictor")
    st.markdown("🏥 **Predict 30-day readmission risk** using patient visit data")
    
    st.markdown("### About the Model")
    st.info("Balanced Random Forest trained on synthetic hospital data.")
    st.caption("Note: For demonstration purposes only.")
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [Download Sample CSV](#)")
    st.markdown("- [Model Documentation](#)")
    st.markdown("- [GitHub Repo](#)")

# ─── MAIN PAGE ───────────────────────────────────────────────────────────
st.title("🏥 30-Day Hospital Readmission Risk Predictor")
st.markdown("Upload patient visit data (raw or processed) to get personalized readmission risk scores.")

# File upload area with icon
uploaded_file = st.file_uploader(
    "**Upload CSV File**",
    type=["csv"],
    help="Supported formats: Raw procedure data or pre-processed features",
    accept_multiple_files=False
)

if uploaded_file is not None:
    with st.spinner("Processing your data..."):
        try:
            df = pd.read_csv(uploaded_file)
            
            # Auto-detect input type
            if 'procedure_datetime' in df.columns:
                st.success("Raw procedure data detected → full preprocessing applied")
                X_pred = preprocess_for_prediction(df)
            elif all(f in df.columns for f in FEATURES):
                st.success("Processed features detected → using directly")
                X_pred = df[FEATURES]
            else:
                st.error("Unsupported format. Please use raw or correctly processed CSV.")
                st.stop()
            
            # Load & predict
            model = load_model()
            probs = model.predict_proba(X_pred)[:, 1]
            
            # Build results
            results = pd.DataFrame({
                'readmission_probability': probs,
                'risk_level': pd.cut(probs, bins=[0, 0.3, 0.5, 1], labels=['Low', 'Medium', 'High']),
                'risk_flag': (probs >= 0.3).astype(int)
            })
            
            if 'patient_id' in df.columns:
                results['patient_id'] = df['patient_id'].values
            if 'visit_id' in df.columns:
                results['visit_id'] = df['visit_id'].values
            
            # ─── DASHBOARD LAYOUT ────────────────────────────────────────
            st.success(f"Processed **{len(results)}** visits successfully!")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Risk Patients", (results['risk_level'] == 'High').sum(), delta_color="inverse")
            with col2:
                st.metric("Medium Risk", (results['risk_level'] == 'Medium').sum())
            with col3:
                st.metric("Average Risk", f"{probs.mean():.1%}")
            with col4:
                st.metric("Highest Risk", f"{probs.max():.1%}")
            
            # Interactive risk distribution chart
            st.subheader("Risk Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(probs, bins=25, kde=True, color='teal', ax=ax)
            ax.axvline(0.3, color='orange', linestyle='--', label='Medium Risk')
            ax.axvline(0.5, color='red', linestyle='--', label='High Risk')
            ax.set_title("Predicted Readmission Probabilities")
            ax.set_xlabel("Probability")
            ax.legend()
            st.pyplot(fig)
            
            # Results table with conditional formatting
            st.subheader("Detailed Predictions")
            def color_risk(val):
                color = 'lightgreen' if val < 0.3 else 'orange' if val < 0.5 else 'salmon'
                return f'background-color: {color}'
            
            styled = results.style.applymap(color_risk, subset=['readmission_probability'])
            st.dataframe(styled)
            
            # Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "readmission_risk_predictions.csv",
                "text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.expander("Show details"):
                st.code(e)

else:
    # Welcome / instructions screen
    st.info("Upload a CSV file to start predicting readmission risks.")
    
    with st.expander("📋 Supported Formats & Example"):
        st.write("**Option 1 – Raw Data** (recommended):")
        st.write("- visit_id, patient_id, procedure_datetime, procedure_code, age, sex, known_chronic_condition")
        
        st.write("**Option 2 – Pre-processed Features**:")
        st.write(FEATURES)
        
        # Show small sample preview
        sample = pd.DataFrame({
            'visit_id': ['V001', 'V002'],
            'patient_id': ['P001', 'P002'],
            'procedure_datetime': ['2023-01-15 09:00', '2023-02-10 14:30'],
            'age': [68, 45],
            'sex': ['M', 'F'],
            'known_chronic_condition': [1, 0]
        })
        st.dataframe(sample)

st.markdown("---")
st.caption("🏥 Built with ❤️ | Model: Balanced Random Forest | Demo only – not for clinical use")
