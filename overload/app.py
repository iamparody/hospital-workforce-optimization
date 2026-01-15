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

def preprocess_for_prediction(df):
    """Minimal preprocessing needed for new prediction data"""
    df['sex_num'] = df['sex'].map({'M': 0, 'F': 1}).fillna(0.5)
    df['chronic_condition_num'] = df['chronic_condition'].astype(int)
    # Assume other features are already numeric and prepared
    return df[FEATURES]

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