# ────────────────────────────────────────────────────────────────
# Combined Healthcare Analytics Dashboard
# ────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Healthcare Analytics Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2, h3 {
        color: #1e40af;
    }
    
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
    
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
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
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_staffing_models():
    """Load workforce optimization models"""
    try:
        with open('healthcare_models.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_readmission_model():
    """Load readmission prediction model"""
    try:
        with open('balanced_random_forest_readmission.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load models at startup
staffing_package = load_staffing_models()
readmission_model = load_readmission_model()

# ═══════════════════════════════════════════════════════════════
# STAFFING ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def preprocess_staffing_data(df):
    """Convert raw CSV to aggregated format for staffing"""
    if 'context_datetime' in df.columns:
        df['context_datetime'] = pd.to_datetime(df['context_datetime'])
        df['date'] = pd.to_datetime(df['context_datetime'].dt.date)
    elif 'date' not in df.columns:
        st.error("❌ CSV must contain 'date' or 'context_datetime' column")
        st.stop()
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    required_cols = ['occupancy_rate', 'staff_per_occupied_bed', 'total_staff']
    if all(col in df.columns for col in required_cols):
        return df
    
    agg_df = df.groupby(['facility_id', 'department_id', 'date']).agg({
        'staff_on_duty': 'sum',
        'nurses_on_duty': 'sum',
        'doctors_on_duty': 'sum',
        'beds_occupied': 'sum',
        'beds_available': 'max'
    }).reset_index()
    
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

def predict_understaffing(agg_df, models, baseline_stats, config):
    """Predict understaffing risk"""
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

def analyze_sustained_overload(agg_df, config):
    """Detect sustained overload patterns"""
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
# READMISSION PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

READMISSION_FEATURES = [
    'age', 'sex_num', 'chronic_condition_num', 'num_procedures',
    'los_days', 'num_prior_visits', 'prev_los', 'prev_readmitted'
]

def preprocess_readmission_data(raw_df):
    """Full preprocessing pipeline for readmission prediction"""
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
    los_probs = los_probs / los_probs.sum()
    
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
    visits['prev_los'] = visits.groupby('patient_id')['los_days'].shift(1).fillna(0)
    visits['prev_readmitted'] = 0
    
    visits['sex_num'] = visits['sex'].map({'M': 0, 'F': 1}).fillna(0.5)
    visits['chronic_condition_num'] = visits['chronic_condition'].astype(int)
    
    return visits[READMISSION_FEATURES]

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🏥 Healthcare Analytics")
    st.markdown("---")
    
    # Module selector
    module = st.radio(
        "Select Module",
        ["👥 Staffing Analytics", "🔄 Readmission Predictor"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Module-specific uploads and settings
    if module == "👥 Staffing Analytics":
        st.markdown("### 📤 Upload Data")
        
        with st.expander("📋 Data Format Guide", expanded=False):
            st.markdown("""
            **Required columns:**
            - `context_datetime` or `date`
            - `facility_id`, `department_id`
            - `staff_on_duty`, `nurses_on_duty`
            - `doctors_on_duty`
            - `beds_occupied`, `beds_available`
            """)
        
        staffing_file = st.file_uploader(
            "Upload Staffing CSV",
            type=['csv'],
            key="staffing_upload",
            help="Upload staffing context data"
        )
        
        if staffing_package:
            st.markdown("---")
            st.markdown("### ⚙️ Model Info")
            config = staffing_package['config']
            baseline_stats = staffing_package['baseline_stats']
            
            st.info(f"""
            **Forecast:** {config['FORECAST_DAYS']} days  
            **Departments:** {len(baseline_stats)}  
            **Status:** ✅ Active
            """)
    
    else:  # Readmission Predictor
        st.markdown("### 📤 Upload Data")
        
        with st.expander("📋 Data Format Guide", expanded=False):
            st.markdown("""
            **Option 1: Raw data**
            - `visit_id`, `patient_id`
            - `procedure_datetime`
            - `procedure_code`
            - `age`, `sex`
            - `known_chronic_condition`
            
            **Option 2: Processed**
            - All engineered features
            """)
        
        readmission_file = st.file_uploader(
            "Upload Patient CSV",
            type=['csv'],
            key="readmission_upload",
            help="Upload patient visit data"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        risk_threshold = st.slider(
            "High Risk Threshold (%)",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            help="Adjust probability threshold"
        )
        
        if readmission_model:
            st.markdown("---")
            st.info("""
            **Algorithm:** Random Forest  
            **Purpose:** 30-Day Risk  
            **Status:** ✅ Active
            """)

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

st.title("🏥 Healthcare Analytics Suite")

# ───────────────────────────────────────────────────────────────
# STAFFING ANALYTICS MODULE
# ───────────────────────────────────────────────────────────────

if module == "👥 Staffing Analytics":
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;'>
        <p style='font-size: 16px; color: #475569; margin: 0;'>
            📊 Predict understaffing risk and detect sustained overload patterns across departments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not staffing_package:
        st.error("⚠️ Staffing models not found. Please ensure `healthcare_models.pkl` is available.")
        st.stop()
    
    if staffing_file:
        with st.spinner("🔄 Processing staffing data..."):
            try:
                raw_df = pd.read_csv(staffing_file)
                agg_df = preprocess_staffing_data(raw_df)
                
                models = staffing_package['models']
                baseline_stats = staffing_package['baseline_stats']
                config = staffing_package['config']
                
                st.success(f"✓ Loaded {len(agg_df):,} records from {agg_df['date'].min().date()} to {agg_df['date'].max().date()}")
                
                risk_df = predict_understaffing(agg_df, models, baseline_stats, config)
                overload_df = analyze_sustained_overload(agg_df, config)
                
                # Display in tabs
                tab1, tab2, tab3 = st.tabs([
                    "📊 Understaffing Risk",
                    "⚠️ Sustained Overload",
                    "📋 Executive Summary"
                ])
                
                with tab1:
                    st.subheader("Departments at Risk (Next 7 Days)")
                    
                    high_risk = risk_df[risk_df['risk_score'] > 0]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("High Risk Departments", len(high_risk), 
                               delta=f"{len(high_risk)}/{len(risk_df)}")
                    col2.metric("Highest Risk Score", f"{risk_df['risk_score'].max():.2f}")
                    col3.metric("Average Risk", f"{risk_df['risk_score'].mean():.2f}")
                    
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
                    
                    st.dataframe(risk_df, use_container_width=True)
                
                with tab2:
                    st.subheader("Historical Sustained Overload Analysis")
                    
                    sustained = overload_df[overload_df['max_consecutive_days'] >= config['MIN_CONSECUTIVE_DAYS']]
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Sustained Overload Departments", len(sustained))
                    col2.metric("Longest Streak", f"{overload_df['max_consecutive_days'].max()} days")
                    
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
                        annotation_text=f"Threshold ({config['MIN_CONSECUTIVE_DAYS']} days)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(overload_df, use_container_width=True)
                
                with tab3:
                    st.subheader("Executive Summary")
                    
                    st.markdown("### 📊 Understaffing Risk")
                    if len(high_risk) > 0:
                        for _, row in high_risk.iterrows():
                            with st.expander(f"🔴 {row['department_id']} - Risk: {row['risk_score']:.2f}"):
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
                    else:
                        st.success("✅ No departments at risk")
                    
                    st.markdown("### ⚠️ Sustained Overload")
                    if len(sustained) > 0:
                        for _, row in sustained.iterrows():
                            with st.expander(f"🔴 {row['department_id']}"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Longest Streak", f"{row['max_consecutive_days']} days")
                                col2.metric("Total Overload Days", row['total_overload_days'])
                                col3.metric("Overload %", f"{row['overload_%']}%")
                    else:
                        st.success("✅ No sustained overload detected")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show details"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload staffing data in the sidebar to begin analysis")

# ───────────────────────────────────────────────────────────────
# READMISSION PREDICTOR MODULE
# ───────────────────────────────────────────────────────────────

else:  # Readmission Predictor
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;'>
        <p style='font-size: 16px; color: #475569; margin: 0;'>
            🔄 Predict 30-day hospital readmission risk using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not readmission_model:
        st.error("⚠️ Readmission model not found. Please ensure `balanced_random_forest_readmission.pkl` is available.")
        st.stop()
    
    if readmission_file:
        with st.spinner("🔄 Processing patient data..."):
            try:
                df = pd.read_csv(readmission_file)
                
                if 'procedure_datetime' in df.columns:
                    st.info("🔍 Processing raw procedure data...")
                    X_pred = preprocess_readmission_data(df)
                elif all(feat in df.columns for feat in READMISSION_FEATURES):
                    st.info("✅ Using processed features")
                    X_pred = df[READMISSION_FEATURES]
                else:
                    st.error("❌ Data format not recognized")
                    st.write("**Available:**", list(df.columns))
                    st.write("**Required:**", READMISSION_FEATURES)
                    st.stop()
                
                probabilities = readmission_model.predict_proba(X_pred)[:, 1]
                
                results_df = pd.DataFrame({
                    'readmission_probability': probabilities,
                    'readmission_risk': (probabilities >= (risk_threshold/100)).astype(int)
                })
                
                if 'patient_id' in df.columns and len(df) == len(results_df):
                    results_df['patient_id'] = df['patient_id'].values
                if 'visit_id' in df.columns and len(df) == len(results_df):
                    results_df['visit_id'] = df['visit_id'].values
                
                st.success(f"✅ Processed **{len(results_df)}** visits")
                
                # Metrics
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
                        f"{high_risk/len(probabilities)*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "🟡 Medium Risk",
                        f"{medium_risk}",
                        f"{medium_risk/len(probabilities)*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "🟢 Low Risk",
                        f"{low_risk}",
                        f"{low_risk/len(probabilities)*100:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "📊 Average Risk",
                        f"{avg_risk:.1%}"
                    )
                
                # Charts
                st.markdown("### 📊 Interactive Visualizations")
                
                tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🎯 Risk Categories", "🔍 Details"])
                
                with tab1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=probabilities,
                        nbinsx=30,
                        marker=dict(
                            color=probabilities,
                            colorscale='RdYlGn_r',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='Probability: %{x:.2%}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_hist.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                                      annotation_text="30% Threshold")
                    fig_hist.add_vline(x=risk_threshold/100, line_dash="dash", line_color="red",
                                      annotation_text=f"{risk_threshold}% High Risk")
                    
                    fig_hist.update_layout(
                        title="Distribution of Readmission Probabilities",
                        xaxis_title="Readmission Probability",
                        yaxis_title="Number of Patients",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with tab2:
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
                        textposition='outside'
                    )])
                    
                    fig_pie.update_layout(
                        title="Risk Category Distribution",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
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
                
                # Results table
                st.markdown("### 📋 Detailed Prediction Results")
                
                display_results = results_df.copy()
                display_results['risk_category'] = pd.cut(
                    display_results['readmission_probability'],
                    bins=[-0.01, 0.3, risk_threshold/100, 1.01],
                    labels=['🟢 Low', '🟡 Medium', '🔴 High']
                )
                display_results['readmission_probability'] = display_results['readmission_probability'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_results, use_container_width=True, height=400)
                
                # Downloads
                st.markdown("### 💾 Export Results")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv,
                        file_name="readmission_predictions.csv",
                        mime="text/csv"
                    )
                
                with col_b:
                    high_risk_df = results_df[results_df['readmission_probability'] >= (risk_threshold/100)]
                    high_risk_csv = high_risk_df.to_csv(index=False)
                    st.download_button(
                        label="⚠️ Download High Risk Only (CSV)",
                        data=high_risk_csv,
                        file_name="high_risk_patients.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.info("👆 Upload patient data in the sidebar to begin prediction")
        
        # Sample data
        st.markdown("### 📝 Sample Data Template")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #1e40af; margin-top: 0;'>📁 Raw Procedure Data</h4>
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
                <h4 style='color: #1e40af; margin-top: 0;'>⚙️ Processed Features</h4>
                <ul style='color: #475569;'>
                    <li><code>age, sex_num, chronic_condition_num</code></li>
                    <li><code>num_procedures, los_days</code></li>
                    <li><code>num_prior_visits, prev_los</code></li>
                    <li><code>prev_readmitted</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
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

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p><strong>🏥 Healthcare Analytics Suite</strong></p>
    <p>Staffing Optimization & Readmission Prediction | For demonstration purposes only</p>
    <p style='font-size: 12px;'>⚠️ Not for clinical decision-making without validation</p>
</div>
""", unsafe_allow_html=True)
