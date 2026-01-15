# app.py - Streamlit app for 30-day readmission prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
    
    # Final dates
    visits['discharge_datetime'] = visits['admission_datetime'] + pd.to_timedelta(visits['los_days'], unit='D')
    visits['admission_date'] = visits['admission_datetime'].dt.date
    visits['discharge_date'] = visits['discharge_datetime'].dt.date
    
    # Sort
    visits = visits.sort_values(['patient_id', 'admission_datetime']).reset_index(drop=True)
    
    # Add required engineered features (minimal version)
    visits['num_prior_visits'] = visits.groupby('patient_id').cumcount()
    visits['prev_readmitted'] = 0  # cannot compute without label, set to 0
    visits['prev_los'] = 0         # cannot compute previous LOS without history, set to 0
    
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

            # Basic validation
            missing_cols = [col for col in ['age', 'sex', 'chronic_condition', 'num_procedures', 
                                           'los_days', 'num_prior_visits', 'prev_los', 'prev_readmitted']
                            if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Preprocess
                X_pred = preprocess_for_prediction(df)

                # Load model
                model = load_model()

                # Predict
                probabilities = model.predict_proba(X_pred)[:, 1]  # probability of class 1 (readmission)

                # Add results to dataframe
                df['readmission_probability'] = probabilities
                df['readmission_risk'] = (probabilities >= 0.5).astype(int)  # threshold 50%

                st.success("Prediction complete!")

                # Show results
                st.subheader("Prediction Results")
                st.dataframe(df[['age', 'sex', 'chronic_condition', 'los_days', 
                                 'readmission_probability', 'readmission_risk']])

                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="readmission_predictions.csv",
                    mime="text/csv"
                )

                # Risk distribution
                st.subheader("Risk Distribution")
                fig, ax = plt.subplots()
                df['readmission_probability'].hist(bins=20, ax=ax)
                ax.set_title("Distribution of Predicted Readmission Probabilities")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:
    st.info("Please upload a CSV file containing the required columns.")

st.markdown("---")
st.caption("Model: Balanced Random Forest | Trained on synthetic data | For demonstration only")
