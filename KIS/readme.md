# 🏥 Hospital Readmission Analytics Dashboard

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Interactive analytics dashboard for predicting and analyzing hospital readmissions within 30 days

---

## 📊 Overview

This dashboard analyzes **2,170 hospital admissions** to identify readmission patterns, predict high-risk patients, and provide actionable insights for reducing preventable readmissions.

**Key Findings:**
- 📈 **3.04% overall readmission rate** (66 cases)
- 🚨 **42.4% readmit within 72 hours** - indicates premature discharge
- 👶 **Pediatric patients highest risk** - 6.3% readmission rate
- ⏱️ **Medium-stay patients (8-14 days)** - 10% readmission rate

---

## 🔍 Data Acquisition

### Source Tables (MySQL Database)
```sql
-- Three main queries executed:
1. Admission-level data (2,170 rows)
   - inpatient_admissions + discharges + patient demographics
   
2. Patient-level aggregations (1,687 unique patients)
   - Total admissions, readmissions, revolving door flags
   
3. Admission requests (176 rows)
   - Request-to-admission workflow analysis
```

### Key Joins
- **Self-join** on `patient_id` to identify readmissions (next admission within 30 days)
- Demographics from `reception_patients`
- Insurance from `reception_patient_schemes`
- Clinical details from `inpatient_discharge_requests`

---

## 🛠️ Feature Engineering

### 1. **Age Group** 
```python
bins: [0, 18, 35, 55, 70, 120]
labels: ['Pediatric', 'Young Adult', 'Middle Age', 'Senior', 'Elderly']
```
**Why:** Age-stratified risk analysis (pediatric 6.3% vs elderly 1.1%)

---

### 2. **Length of Stay (LOS) Category**
```python
bins: [0, 2, 7, 14, 999]
labels: ['Very Short (<2d)', 'Short (2-7d)', 'Medium (8-14d)', 'Long (15+d)']
```
**Why:** Medium stays show 10% readmission rate - clinical red flag

---

### 3. **Cost Category**
```python
bins: [0, 6000, 8000, 15000]
labels: ['Low', 'Medium', 'High']
```
**Why:** Financial impact segmentation ($456K spent on readmissions)

---

### 4. **Risk Score** ⭐ **(Most Important)**
```python
Risk Score = 
  + (Previous Admissions × 10)      # History of frequent visits
  + (Age >65 OR Age <5) × 15        # Vulnerable populations
  + (LOS <2 days) × 10              # Potential premature discharge
  + (Bad Discharge Type) × 20       # Against medical advice/Absconded
```

**Interpretation:**
- **0-20:** Low risk 🟢
- **20-40:** Medium risk 🟡  
- **40+:** High risk 🔴 (requires intervention)

**Purpose:** Predict which patients need post-discharge follow-up

---

### 5. **Previous Admission Category**
```python
bins: [0, 1, 3, 6, 999]
labels: ['First Time', '1-2 Prior', '3-5 Prior', '6+ Prior']
```
**Why:** Repeat patients need different care coordination strategies

---

### 6. **Revolving Door Flag** 🔄
```python
Definition: Patient with 3+ admissions in 6 months
Calculation: Flagged in SQL, mapped to admission records
```

**What it means:**  
"Frequent flyer" patients who repeatedly return to hospital

**Why it matters:**  
- Indicates care gaps or chronic conditions
- Requires case management and care coordination
- Breaking the cycle reduces costs and improves outcomes

---

### 7. **Days to Readmission Category**
```python
bins: [0, 3, 7, 14, 30]
labels: ['0-3 days', '4-7 days', '8-14 days', '15-30 days']
```
**Why:** 72-hour readmissions (0-3 days) are quality metric - often preventable

---

## 📈 Key Metrics

| Metric | Value | Significance |
|--------|-------|--------------|
| **Total Admissions** | 2,170 | Dataset size |
| **30-Day Readmission Rate** | 3.04% | Primary outcome measure |
| **72-Hour Readmissions** | 42.4% of readmissions | Critical quality indicator |
| **High-Risk Patients** | 0.6% | Require immediate intervention |
| **Readmission Cost** | $456K (3.1% of total) | Financial impact |
| **Potential Savings** | $228K (50% reduction) | ROI opportunity |

---

## 🎯 Dashboard Features

### **6 Interactive Tabs:**
1. **📊 Overview** - KPIs, trends, demographics
2. **🔄 Readmission Analysis** - Patterns by age, discharge type, timing
3. **⏱️ Length of Stay** - LOS impact on outcomes
4. **⚠️ Risk Analytics** - Risk scoring, high-risk patient list
5. **💰 Financial Impact** - Cost analysis, ROI calculator
6. **📝 Insights Summary** - Actionable recommendations

### **Interactive Filters:**
- Date range
- Age group
- Discharge type
- Payment mode

---

## 🚀 Quick Start

```bash
# Clone repository
git clone <repo-url>
cd hospital-readmission-dashboard

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run readmission_dashboard.py
```

**Access:** `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
numpy>=1.24.0
```

---

## 🔑 Key Insights

### 🔴 **Critical Findings**
1. **Premature Discharge Crisis** - 42.4% readmit within 72 hours
2. **Pediatric High Risk** - 6.3% rate vs 1.1% for elderly
3. **Medium-Stay Paradox** - 8-14 day stays have 10% readmission rate
4. **Financial Burden** - $456K in preventable costs

### 💡 **Recommended Actions**
- Implement discharge readiness assessments
- 72-hour follow-up calls for all patients
- Enhanced protocols for pediatric cases
- Risk-stratified discharge planning (using risk scores)
- Case management for high-risk patients (score >40)

---

## 📄 License

MIT License - feel free to use and modify

---

## 👤 Author

Data Science Portfolio Project | 2025

---

**Built with:** Python • Streamlit • Plotly • Pandas
