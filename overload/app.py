# app.py - Enhanced Streamlit app for 30-day readmission prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# ─── CONFIGURATION ───────────────────────────────────────────────────────
MODEL_PATH = "overload/balanced_random_forest_readmission.pkl"
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
    visits['num_prior_visits'] = visits.groupby('patient_id').cumcount()
    visits['prev_los'] = visits.groupby('patient_id')['los_days'].shift(1).fillna(0)
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
        st.error(f"🚨 Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        st.stop()

# ─── STREAMLIT APP ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="30-Day Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2, h3 {
        color: #1e40af;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1e40af;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏥 Readmission Predictor")
    st.markdown("---")
    
    st.markdown("### 📊 Model Information")
    st.info("""
    **Algorithm:** Balanced Random Forest  
    **Purpose:** 30-Day Readmission Risk  
    **Status:** ✅ Active
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    risk_threshold = st.slider(
        "High Risk Threshold (%)",
        min_value=30,
        max_value=70,
        value=50,
        step=5,
        help="Adjust the probability threshold for high-risk classification"
    )
    
    show_details = st.toggle("Show detailed statistics", value=True)
    
    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    1. **Upload** your patient data CSV
    2. **Review** predictions and risk levels
    3. **Download** results for your records
    4. **Analyze** trends in the charts
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool predicts hospital readmission risk using machine learning.
    
    **Version:** 2.0  
    **Last Updated:** Jan 2025
    """)

# ─── MAIN CONTENT ────────────────────────────────────────────────────────
st.title("🏥 30-Day Hospital Readmission Risk Predictor")
st.markdown("""
<div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;'>
    <p style='font-size: 16px; color: #475569; margin: 0;'>
        📁 Upload patient visit data to predict readmission risk within 30 days using our advanced ML model.
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload Your CSV File",
    type=["csv"],
    help="Upload patient visit data in CSV format"
)

if uploaded_file is not None:
    with st.spinner("🔄 Processing your data..."):
        try:
            # Read uploaded data
            df = pd.read_csv(uploaded_file)
            
            # Check if this is raw data or already processed
            if 'procedure_datetime' in df.columns:
                st.info("🔍 Detected raw procedure data. Processing to create features...")
                X_pred = preprocess_for_prediction(df)
                
            elif all(feat in df.columns for feat in FEATURES):
                st.info("✅ Detected already processed data with engineered features.")
                X_pred = df[FEATURES]
                
            else:
                st.error("❌ Data format not recognized.")
                st.write("**Available columns in your file:**")
                st.write(list(df.columns))
                st.write("**Required for raw data:** `visit_id`, `patient_id`, `procedure_datetime`, `procedure_code`, `age`, `sex`, `known_chronic_condition`")
                st.write("**Required for processed data:**", FEATURES)
                st.stop()
            
            # Load model
            model = load_model()
            
            # Predict
            probabilities = model.predict_proba(X_pred)[:, 1]
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'readmission_probability': probabilities,
                'readmission_risk': (probabilities >= (risk_threshold/100)).astype(int)
            })
            
            # Add patient ID and visit ID if available
            if 'patient_id' in df.columns and len(df) == len(results_df):
                results_df['patient_id'] = df['patient_id'].values
            if 'visit_id' in df.columns and len(df) == len(results_df):
                results_df['visit_id'] = df['visit_id'].values
            
            st.success(f"✅ Prediction complete! Processed **{len(results_df)}** visits.")
            
            # ─── METRICS DASHBOARD ───────────────────────────────────────
            st.markdown("### 📈 Risk Summary Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk = (probabilities >= (risk_threshold/100)).sum()
            medium_risk = ((probabilities >= 0.3) & (probabilities < (risk_threshold/100))).sum()
            low_risk = (probabilities < 0.3).sum()
            avg_risk = probabilities.mean()
            
            with col1:
                st.metric(
                    "🔴 High Risk",
                    f"{high_risk}",
                    f"{high_risk/len(probabilities)*100:.1f}%",
                    help=f"Patients with ≥{risk_threshold}% readmission probability"
                )
            
            with col2:
                st.metric(
                    "🟡 Medium Risk",
                    f"{medium_risk}",
                    f"{medium_risk/len(probabilities)*100:.1f}%",
                    help=f"Patients with 30-{risk_threshold}% probability"
                )
            
            with col3:
                st.metric(
                    "🟢 Low Risk",
                    f"{low_risk}",
                    f"{low_risk/len(probabilities)*100:.1f}%",
                    help="Patients with <30% readmission probability"
                )
            
            with col4:
                st.metric(
                    "📊 Average Risk",
                    f"{avg_risk:.1%}",
                    help="Mean readmission probability across all patients"
                )
            
            # ─── INTERACTIVE CHARTS ──────────────────────────────────────
            st.markdown("### 📊 Interactive Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🎯 Risk Categories", "🔍 Details"])
            
            with tab1:
                # Histogram with Plotly
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=probabilities,
                    nbinsx=30,
                    marker=dict(
                        color=probabilities,
                        colorscale='RdYlGn_r',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Probability: %{x:.2%}<br>Count: %{y}<extra></extra>',
                    name='Distribution'
                ))
                
                # Add threshold lines
                fig_hist.add_vline(
                    x=0.3,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="30% Threshold",
                    annotation_position="top"
                )
                
                fig_hist.add_vline(
                    x=risk_threshold/100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{risk_threshold}% High Risk",
                    annotation_position="top"
                )
                
                fig_hist.update_layout(
                    title="Distribution of Readmission Probabilities",
                    xaxis_title="Readmission Probability",
                    yaxis_title="Number of Patients",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with tab2:
                # Pie chart for risk categories
                risk_categories = pd.DataFrame({
                    'Category': ['Low Risk', 'Medium Risk', 'High Risk'],
                    'Count': [low_risk, medium_risk, high_risk],
                    'Color': ['#10b981', '#f59e0b', '#ef4444']
                })
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_categories['Category'],
                    values=risk_categories['Count'],
                    marker=dict(colors=risk_categories['Color']),
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside',
                    hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    title="Risk Category Distribution",
                    template="plotly_white",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab3:
                # Box plot
                results_copy = results_df.copy()
                results_copy['Risk Category'] = pd.cut(
                    results_copy['readmission_probability'],
                    bins=[-0.01, 0.3, risk_threshold/100, 1.01],
                    labels=['Low', 'Medium', 'High']
                )
                
                fig_box = px.box(
                    results_copy,
                    x='Risk Category',
                    y='readmission_probability',
                    color='Risk Category',
                    color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'},
                    title="Risk Probability by Category"
                )
                
                fig_box.update_layout(
                    yaxis_title="Readmission Probability",
                    template="plotly_white",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            # ─── RESULTS TABLE ───────────────────────────────────────────
            st.markdown("### 📋 Detailed Prediction Results")
            
            # Add risk category to display
            display_results = results_df.copy()
            display_results['risk_category'] = pd.cut(
                display_results['readmission_probability'],
                bins=[-0.01, 0.3, risk_threshold/100, 1.01],
                labels=['🟢 Low', '🟡 Medium', '🔴 High']
            )
            display_results['readmission_probability'] = display_results['readmission_probability'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                display_results,
                use_container_width=True,
                height=400
            )
            
            # ─── DOWNLOAD SECTION ────────────────────────────────────────
            st.markdown("### 💾 Export Results")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="readmission_predictions.csv",
                    mime="text/csv",
                    help="Download complete predictions with all columns"
                )
            
            with col_b:
                high_risk_df = results_df[results_df['readmission_probability'] >= (risk_threshold/100)]
                high_risk_csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="⚠️ Download High Risk Only (CSV)",
                    data=high_risk_csv,
                    file_name="high_risk_patients.csv",
                    mime="text/csv",
                    help="Download only high-risk patient predictions"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Show error details"):
                import traceback
                st.code(traceback.format_exc())

else:
    # Instructions when no file is uploaded
    st.markdown("### 📋 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4 style='color: #1e40af; margin-top: 0;'>📁 Option 1: Raw Procedure Data</h4>
            <p style='color: #64748b;'>Upload procedure-level data with these columns:</p>
            <ul style='color: #475569;'>
                <li><code>visit_id</code> - Visit identifier</li>
                <li><code>patient_id</code> - Patient identifier</li>
                <li><code>procedure_datetime</code> - Date/time</li>
                <li><code>procedure_code</code> - Procedure code</li>
                <li><code>age</code> - Patient age</li>
                <li><code>sex</code> - Gender (M/F)</li>
                <li><code>known_chronic_condition</code> - 0 or 1</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4 style='color: #1e40af; margin-top: 0;'>⚙️ Option 2: Processed Features</h4>
            <p style='color: #64748b;'>Upload pre-processed data with engineered features:</p>
            <ul style='color: #475569;'>
                <li><code>age, sex_num, chronic_condition_num</code></li>
                <li><code>num_procedures, los_days</code></li>
                <li><code>num_prior_visits, prev_los</code></li>
                <li><code>prev_readmitted</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data preview
    st.markdown("### 📝 Sample Data Template")
    
    sample_data = pd.DataFrame({
        'visit_id': ['V1001', 'V1001', 'V1002'],
        'patient_id': ['P001', 'P001', 'P002'],
        'procedure_datetime': ['2023-01-15 09:30', '2023-01-15 11:00', '2023-01-16 10:00'],
        'procedure_code': ['CPT-001', 'CPT-002', 'CPT-001'],
        'age': [65, 65, 72],
        'sex': ['M', 'M', 'F'],
        'known_chronic_condition': [1, 1, 0]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        "📥 Download Sample Template",
        data=csv_sample,
        file_name="sample_readmission_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p><strong>🏥 Hospital Readmission Risk Predictor</strong></p>
    <p>Powered by Balanced Random Forest ML Model | For demonstration purposes only</p>
    <p style='font-size: 12px;'>⚠️ Not for clinical decision-making without validation</p>
</div>
""", unsafe_allow_html=True)
