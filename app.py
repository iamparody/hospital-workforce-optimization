import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


st.sidebar.radio("Theme", ["Light", "Dark"], key="theme")
# Page config
st.set_page_config(
    page_title="Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stMetric label {font-size: 14px !important; font-weight: 600 !important;}
    .stMetric [data-testid="stMetricValue"] {font-size: 32px !important;}
    h1 {color: #1f2937; font-weight: 700;}
    h2 {color: #374151; font-weight: 600; font-size: 24px;}
    h3 {color: #4b5563; font-weight: 600; font-size: 18px;}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('readmission_powerbi_final.csv')
    df['admitted_at'] = pd.to_datetime(df['admitted_at'])
    df['discharged_at'] = pd.to_datetime(df['discharged_at'])
    df['month'] = df['admitted_at'].dt.to_period('M').astype(str)
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.title("🔍 Filters")
    
    # Date range
    min_date = df['admitted_at'].min().date()
    max_date = df['admitted_at'].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Age group
    age_groups = st.multiselect("Age Group", options=df['age_group'].dropna().unique(), default=df['age_group'].dropna().unique())
    
    # Discharge type
    discharge_types = st.multiselect("Discharge Type", options=df['discharge_type_name'].unique(), default=df['discharge_type_name'].unique())
    
    # Payment mode
    payment_modes = st.multiselect("Payment Mode", options=df['payment_mode'].unique(), default=df['payment_mode'].unique())
    
    # Apply filters
    if len(date_range) == 2:
        mask = (
            (df['admitted_at'].dt.date >= date_range[0]) & 
            (df['admitted_at'].dt.date <= date_range[1]) &
            (df['age_group'].isin(age_groups)) &
            (df['discharge_type_name'].isin(discharge_types)) &
            (df['payment_mode'].isin(payment_modes))
        )
        filtered_df = df[mask]
    else:
        filtered_df = df

# Title
st.title("🏥 Hospital Readmission Analytics Dashboard")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🔄 Readmission Analysis", 
    "⏱️ Length of Stay", 
    "⚠️ Risk Analytics", 
    "💰 Financial Impact",
    "📝 Insights Summary"
])

# TAB 1: OVERVIEW
with tab1:
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_admissions = len(filtered_df)
    total_readmissions = filtered_df['is_readmission'].sum()
    readmit_rate = (total_readmissions / total_admissions * 100) if total_admissions > 0 else 0
    readmit_cost = filtered_df[filtered_df['is_readmission']==1]['cost'].sum()
    
    with col1:
        st.metric("Total Admissions", f"{total_admissions:,}", 
                  delta=None, delta_color="normal")
    
    with col2:
        st.metric("Readmission Rate", f"{readmit_rate:.2f}%", 
                  delta=f"{readmit_rate-3.04:.2f}% vs baseline", 
                  delta_color="inverse")
    
    with col3:
        st.metric("Total Readmissions", f"{total_readmissions:,}")
    
    with col4:
        st.metric("Readmission Cost", f"KES{readmit_cost/1000:.0f}K")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Trend
        monthly = filtered_df.groupby('month').agg({
            'admission_id': 'count',
            'is_readmission': 'sum'
        }).reset_index()
        monthly['rate'] = (monthly['is_readmission'] / monthly['admission_id'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['rate'],
            mode='lines+markers',
            name='Readmission Rate',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Readmission Rate Trend",
            xaxis_title="Month",
            yaxis_title="Rate (%)",
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age Distribution
        age_dist = filtered_df['age_group'].value_counts().reset_index()
        age_dist.columns = ['age_group', 'count']
        
        fig = px.pie(age_dist, values='count', names='age_group',
                     title="Admissions by Age Group",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment Mode
        payment_dist = filtered_df['payment_mode'].value_counts().reset_index()
        payment_dist.columns = ['payment_mode', 'count']
        
        fig = px.pie(payment_dist, values='count', names='payment_mode',
                     title="Admissions by Payment Mode",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Discharge Type Distribution
        discharge_dist = filtered_df['discharge_type_name'].value_counts().reset_index()
        discharge_dist.columns = ['discharge_type', 'count']
        
        fig = px.bar(discharge_dist, x='discharge_type', y='count',
                     title="Discharges by Type",
                     color='count',
                     color_continuous_scale='Blues')
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: READMISSION ANALYSIS
with tab2:
    col1, col2, col3 = st.columns(3)
    
    readmitted = filtered_df[filtered_df['is_readmission']==1]
    critical_72hr = (readmitted['days_to_readmission'] <= 3).sum()
    critical_pct = (critical_72hr / len(readmitted) * 100) if len(readmitted) > 0 else 0
    avg_days = readmitted['days_to_readmission'].mean()
    
    with col1:
        st.metric("72-Hour Readmissions", f"{critical_72hr}", 
                  delta=f"{critical_pct:.1f}% of readmissions")
    
    with col2:
        st.metric("Avg Days to Readmission", f"{avg_days:.1f}" if not pd.isna(avg_days) else "N/A")
    
    with col3:
        high_risk = (filtered_df['risk_score'] > 40).sum()
        st.metric("High Risk Patients", f"{high_risk:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission by Age Group
        age_readmit = filtered_df.groupby('age_group').agg({
            'admission_id': 'count',
            'is_readmission': 'sum'
        }).reset_index()
        age_readmit['rate'] = (age_readmit['is_readmission'] / age_readmit['admission_id'] * 100)
        age_readmit = age_readmit.sort_values('rate', ascending=False)
        
        fig = px.bar(age_readmit, x='age_group', y='rate',
                     title="Readmission Rate by Age Group",
                     color='rate',
                     color_continuous_scale='Reds',
                     text='rate')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        fig.add_hline(y=readmit_rate, line_dash="dash", line_color="gray", 
                      annotation_text=f"Overall: {readmit_rate:.2f}%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Readmission by Discharge Type
        discharge_readmit = filtered_df.groupby('discharge_type_name').agg({
            'admission_id': 'count',
            'is_readmission': 'sum'
        }).reset_index()
        discharge_readmit['rate'] = (discharge_readmit['is_readmission'] / discharge_readmit['admission_id'] * 100)
        discharge_readmit = discharge_readmit.sort_values('rate', ascending=True)
        
        fig = px.bar(discharge_readmit, y='discharge_type_name', x='rate',
                     title="Readmission Rate by Discharge Type",
                     orientation='h',
                     color='rate',
                     color_continuous_scale='Oranges',
                     text='rate')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        fig.add_vline(x=readmit_rate, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Days to Readmission Distribution
    if len(readmitted) > 0:
        fig = px.histogram(readmitted, x='days_to_readmission', nbins=15,
                          title="Days to Readmission Distribution",
                          color_discrete_sequence=['#3b82f6'])
        fig.add_vline(x=3, line_dash="dash", line_color="red", 
                      annotation_text="72 hours", annotation_position="top")
        fig.add_vline(x=7, line_dash="dash", line_color="orange", 
                      annotation_text="7 days", annotation_position="top")
        fig.update_layout(height=350, xaxis_title="Days", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: LENGTH OF STAY
with tab3:
    col1, col2, col3 = st.columns(3)
    
    avg_los = filtered_df['length_of_stay'].mean()
    avg_los_readmit = filtered_df[filtered_df['is_readmission']==1]['length_of_stay'].mean()
    max_los = filtered_df['length_of_stay'].max()
    
    with col1:
        st.metric("Avg LOS (All)", f"{avg_los:.1f} days")
    
    with col2:
        st.metric("Avg LOS (Readmissions)", f"{avg_los_readmit:.1f} days" if not pd.isna(avg_los_readmit) else "N/A")
    
    with col3:
        st.metric("Max LOS", f"{max_los} days")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # LOS Category Readmission Rate
        los_readmit = filtered_df.groupby('los_category').agg({
            'admission_id': 'count',
            'is_readmission': 'sum'
        }).reset_index()
        los_readmit['rate'] = (los_readmit['is_readmission'] / los_readmit['admission_id'] * 100)
        
        fig = px.bar(los_readmit, x='los_category', y='rate',
                     title="Readmission Rate by Length of Stay",
                     color='rate',
                     color_continuous_scale='RdYlGn_r',
                     text='rate')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        fig.add_hline(y=readmit_rate, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # LOS Distribution
        fig = px.box(filtered_df, y='length_of_stay', x='age_group',
                     title="LOS Distribution by Age Group",
                     color='age_group',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, showlegend=False, yaxis_title="Days")
        st.plotly_chart(fig, use_container_width=True)
    
    # LOS vs Cost Scatter
    fig = px.scatter(filtered_df.sample(min(500, len(filtered_df))), 
                     x='length_of_stay', y='cost',
                     color='is_readmission',
                     color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
                     labels={'is_readmission': 'Readmitted'},
                     title="Length of Stay vs Cost",
                     opacity=0.6)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: RISK ANALYTICS
with tab4:
    col1, col2, col3 = st.columns(3)
    
    high_risk = (filtered_df['risk_score'] > 40).sum()
    medium_risk = ((filtered_df['risk_score'] > 20) & (filtered_df['risk_score'] <= 40)).sum()
    low_risk = (filtered_df['risk_score'] <= 20).sum()
    
    with col1:
        st.metric("🔴 High Risk (>40)", f"{high_risk:,}", 
                  delta=f"{high_risk/len(filtered_df)*100:.1f}%")
    
    with col2:
        st.metric("🟡 Medium Risk (20-40)", f"{medium_risk:,}",
                  delta=f"{medium_risk/len(filtered_df)*100:.1f}%")
    
    with col3:
        st.metric("🟢 Low Risk (<20)", f"{low_risk:,}",
                  delta=f"{low_risk/len(filtered_df)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Score Distribution
        fig = px.histogram(filtered_df, x='risk_score', nbins=20,
                          title="Risk Score Distribution",
                          color_discrete_sequence=['#8b5cf6'])
        fig.add_vline(x=20, line_dash="dash", line_color="green", 
                      annotation_text="Low Risk")
        fig.add_vline(x=40, line_dash="dash", line_color="red", 
                      annotation_text="High Risk")
        fig.update_layout(height=400, xaxis_title="Risk Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Score vs Actual Readmission
        risk_groups = pd.cut(filtered_df['risk_score'], bins=[0, 20, 40, 100], 
                            labels=['Low', 'Medium', 'High'])
        risk_actual = filtered_df.groupby(risk_groups).agg({
            'admission_id': 'count',
            'is_readmission': 'sum'
        }).reset_index()
        risk_actual.columns = ['risk_group', 'total', 'readmissions']
        risk_actual['rate'] = (risk_actual['readmissions'] / risk_actual['total'] * 100)
        
        fig = px.bar(risk_actual, x='risk_group', y='rate',
                     title="Actual Readmission Rate by Risk Group",
                     color='rate',
                     color_continuous_scale='Reds',
                     text='rate')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_title="Risk Group", 
                         yaxis_title="Readmission Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # High Risk Patients Table
    st.subheader("🚨 Top 20 High-Risk Patients")
    high_risk_patients = filtered_df.nlargest(20, 'risk_score')[
        ['patient_id', 'first_name', 'last_name', 'age_at_admission', 
         'risk_score', 'previous_admission_count', 'discharge_type_name', 'is_readmission']
    ].copy()
    high_risk_patients['is_readmission'] = high_risk_patients['is_readmission'].map({0: '❌', 1: '✅'})
    
    st.dataframe(high_risk_patients, use_container_width=True, height=400)

# TAB 5: FINANCIAL IMPACT
with tab5:
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = filtered_df['cost'].sum()
    readmit_cost = filtered_df[filtered_df['is_readmission']==1]['cost'].sum()
    readmit_cost_pct = (readmit_cost / total_cost * 100) if total_cost > 0 else 0
    avg_cost = filtered_df['cost'].mean()
    
    with col1:
        st.metric("Total Cost", f"KES{total_cost/1000000:.2f}M")
    
    with col2:
        st.metric("Readmission Cost", f"KES{readmit_cost/1000:.0f}K",
                  delta=f"{readmit_cost_pct:.1f}% of total")
    
    with col3:
        st.metric("Avg Cost/Admission", f"KES{avg_cost:,.0f}")
    
    with col4:
        # Potential savings if readmission reduced by 50%
        potential_savings = readmit_cost * 0.5
        st.metric("Potential Savings (50% reduction)", f"KES{potential_savings/1000:.0f}K")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost by Age Group
        cost_age = filtered_df.groupby('age_group')['cost'].agg(['mean', 'sum']).reset_index()
        cost_age.columns = ['age_group', 'avg_cost', 'total_cost']
        
        fig = px.bar(cost_age, x='age_group', y='total_cost',
                     title="Total Cost by Age Group",
                     color='avg_cost',
                     color_continuous_scale='Greens',
                     text='total_cost')
        fig.update_traces(texttemplate='KES%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False,
                         yaxis_title="Total Cost (KES)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost: Initial vs Readmission
        cost_comparison = pd.DataFrame({
            'Type': ['Initial Admission', 'Readmission'],
            'Avg Cost': [
                filtered_df[filtered_df['is_readmission']==0]['cost'].mean(),
                filtered_df[filtered_df['is_readmission']==1]['cost'].mean()
            ]
        })
        
        fig = px.bar(cost_comparison, x='Type', y='Avg Cost',
                     title="Average Cost: Initial vs Readmission",
                     color='Avg Cost',
                     color_continuous_scale='Oranges',
                     text='Avg Cost')
        fig.update_traces(texttemplate='KES%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Cost Trend
    monthly_cost = filtered_df.groupby('month').agg({
        'cost': 'sum',
        'is_readmission': lambda x: filtered_df.loc[x.index, 'cost'][filtered_df.loc[x.index, 'is_readmission']==1].sum()
    }).reset_index()
    monthly_cost.columns = ['month', 'total_cost', 'readmit_cost']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_cost['month'], y=monthly_cost['total_cost'],
                        name='Total Cost', marker_color='#3b82f6'))
    fig.add_trace(go.Bar(x=monthly_cost['month'], y=monthly_cost['readmit_cost'],
                        name='Readmission Cost', marker_color='#ef4444'))
    fig.update_layout(title="Monthly Cost Breakdown", barmode='group', height=350,
                     yaxis_title="Cost (KES)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

# TAB 6: INSIGHTS SUMMARY
with tab6:
    st.header("📝 Key Insights & Recommendations")
    
    # Calculate key metrics
    pediatric_rate = filtered_df[filtered_df['age_group']=='Pediatric (0-17)']['is_readmission'].mean() * 100 if 'Pediatric (0-17)' in filtered_df['age_group'].values else 0
    medium_los_rate = filtered_df[filtered_df['los_category']=='Medium (8-14d)']['is_readmission'].mean() * 100 if 'Medium (8-14d)' in filtered_df['los_category'].values else 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔴 Critical Findings")
        
        st.markdown(f"""
        **1. Premature Discharge Crisis**
        - **{critical_pct:.1f}% of readmissions occur within 72 hours** ({critical_72hr} patients)
        - This suggests patients are being discharged too early
        - **Action:** Implement discharge readiness assessments
        
        **2. Pediatric Population at Risk**
        - Pediatric patients (0-17) have **{pediatric_rate:.1f}% readmission rate** - highest among all age groups
        - 2x higher than elderly patients
        - **Action:** Enhanced post-discharge follow-up for pediatric cases
        
        **3. Medium Length of Stay Paradox**
        - Patients staying 8-14 days have **{medium_los_rate:.1f}% readmission rate**
        - Much higher than short (2-7 days) or long stays (15+ days)
        - **Action:** Review clinical protocols for medium-stay patients
        
        **4. Financial Impact**
        - Readmissions cost **KES{readmit_cost/1000:.0f}K** ({readmit_cost_pct:.1f}% of total costs)
        - Average readmission costs **KES{filtered_df[filtered_df['is_readmission']==1]['cost'].mean():,.0f}**
        - **Potential annual savings: KES{potential_savings/1000:.0f}K** with 50% reduction
        """)
        
        st.markdown("### 🟡 Secondary Observations")
        
        stable_discharge_rate = filtered_df[filtered_df['discharge_type_name']=='Patient is Stable']['is_readmission'].mean() * 100 if 'Patient is Stable' in filtered_df['discharge_type_name'].values else 0
        
        st.markdown(f"""
        **5. Discharge Type Matters**
        - "Patient is Stable" discharges have **{stable_discharge_rate:.1f}% readmission rate**
        - Higher than "Patient Request" discharges
        - Question: Are clinical assessments accurate?
        
        **6. High-Risk Patient Concentration**
        - **{high_risk:,} patients** ({high_risk/len(filtered_df)*100:.1f}%) are high-risk (score >40)
        - These patients need proactive case management
        - Implement risk-stratified discharge planning
        """)
    
    with col2:
        st.markdown("### 🎯 Recommended Actions")
        
        st.info("""
        **Immediate (0-30 days)**
        1. Flag patients discharged <2 days
        2. 72-hour follow-up calls for all
        3. Review pediatric protocols
        
        **Short-term (1-3 months)**
        1. Implement risk scoring at discharge
        2. Enhanced discharge planning for medium-stay patients
        3. Case management for high-risk patients
        
        **Long-term (3-6 months)**
        1. Predictive model deployment
        2. Care coordination program
        3. Post-discharge clinic visits
        4. Patient education materials
        """)
        
        st.success(f"""
        **Expected Impact**
        - 30% readmission reduction
        - **KES{readmit_cost * 0.3 / 1000:.0f}K** annual savings
        - Improved patient outcomes
        - Better quality metrics
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Dashboard Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Coverage**")
        st.markdown(f"- **{total_admissions:,}** total admissions")
        st.markdown(f"- **{filtered_df['patient_id'].nunique():,}** unique patients")
        st.markdown(f"- Date range: {filtered_df['admitted_at'].min().date()} to {filtered_df['admitted_at'].max().date()}")
    
    with col2:
        st.markdown("**Readmission Metrics**")
        st.markdown(f"- Overall rate: **{readmit_rate:.2f}%**")
        st.markdown(f"- Critical (72hr): **{critical_72hr}** cases")
        st.markdown(f"- Avg days to readmit: **{avg_days:.1f}**")
    
    with col3:
        st.markdown("**Risk Distribution**")
        st.markdown(f"- 🔴 High: **{high_risk:,}** ({high_risk/len(filtered_df)*100:.1f}%)")
        st.markdown(f"- 🟡 Medium: **{medium_risk:,}** ({medium_risk/len(filtered_df)*100:.1f}%)")
        st.markdown(f"- 🟢 Low: **{low_risk:,}** ({low_risk/len(filtered_df)*100:.1f}%)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 12px;'>
    Hospital Readmission Analytics Dashboard | Data updated: {} | Built with Streamlit
</div>

""".format(pd.Timestamp.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)
