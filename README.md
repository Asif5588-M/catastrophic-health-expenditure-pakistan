# 🏥 Predicting Catastrophic Health Expenditure in Pakistan

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://asifnawaz-pakistan-health.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![WHO SDG 3.8](https://img.shields.io/badge/WHO-SDG%203.8-009EDB?style=for-the-badge)](https://www.who.int/health-topics/universal-health-coverage)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **An end-to-end Machine Learning project** predicting catastrophic out-of-pocket health expenditure in Pakistan using World Bank, WHO, and clinical datasets — aligned with WHO SDG 3.8 (Universal Health Coverage).

---

## 🔴 The Problem

Pakistan spends **52.9% of its health budget through out-of-pocket payments** — far above the WHO danger threshold of 40%. This means millions of households face **catastrophic health expenditure (CHE)** — spending so much on healthcare that they are pushed into poverty.

| Indicator | Pakistan | WHO Threshold |
|-----------|----------|---------------|
| OOP % of Current Health Expenditure | 52.9% | < 40% |
| Years Above Threshold (2000–2023) | **24 / 24** | — |
| Population facing CHE (10% threshold) | ~6.8% | — |
| Govt Health Spending (% of CHE) | ~1% | — |

---

## 🎯 Project Objectives

- Identify **which households are at highest risk** of catastrophic health expenditure
- Apply **explainable ML (SHAP)** to uncover the key drivers
- Build an **interactive dashboard** for policymakers and researchers
- Contribute to **WHO SDG 3.8** — Universal Health Coverage for Pakistan

---

## 📊 Live Dashboard

**[🚀 Open Dashboard →](https://asifnawaz-pakistan-health.streamlit.app)**

Features:
- Real-time OOP expenditure trend (World Bank 2000–2023)
- WHO catastrophic expenditure visualization
- Interactive income filter — see how risk changes by income group
- SHAP feature importance — what drives catastrophic expenditure
- Severity distribution across 55,500 patient records

---

## 🗂️ Project Structure

```
catastrophic-health-expenditure-pakistan/
│
├── data/
│   └── raw/
│       ├── wb_pakistan_health_indicators.csv   # World Bank data
│       ├── who_pakistan_catastrophic_exp.csv   # WHO GHO data
│       ├── healthcare_dataset.csv              # Clinical dataset
│       ├── ml_ready_dataset.csv                # Feature-engineered ML data
│       └── master_pakistan_health.csv          # Merged master dataset
│
├── notebooks/
│   └── 01_eda.ipynb                            # Full EDA + ML pipeline
│
├── dashboard/
│   └── streamlit_app.py                        # Interactive dashboard
│
├── reports/
│   └── figures/                                # Publication-ready plots
│       ├── 01_wb_health_trends.png
│       ├── 02_billing_analysis.png
│       ├── 03_catastrophic_who_trends.png
│       ├── 06_clean_model_comparison.png
│       └── 07_shap_clean.png
│
├── src/
│   └── download_data.py                        # Automated data pipeline
│
├── requirements.txt
└── README.md
```

---

## 🤖 ML Pipeline

### Data Sources
| Source | Description | Records |
|--------|-------------|---------|
| World Bank API | Pakistan health indicators 2000–2023 | 30 years |
| WHO GHO API | Catastrophic expenditure data | 8 time points |
| Kaggle Healthcare | Clinical patient records | 55,500 |

### Models Trained
| Model | AUC | F1 (Catastrophic) | Accuracy |
|-------|-----|-------------------|----------|
| Logistic Regression | 0.799 | 0.960 | 92.4% |
| **Random Forest** | **0.803** | 0.948 | 90.4% |
| XGBoost | 0.795 | 0.965 | 93.2% |

### SHAP Key Findings
The model reveals that catastrophic health expenditure in Pakistan is driven primarily by **income level — not disease type**:

1. `monthly_income_usd` — SHAP 0.953 *(dominant factor)*
2. `income_low` — SHAP 0.207
3. `Age` — SHAP 0.179
4. `length_of_stay` — SHAP 0.158
5. `insurer_enc` — SHAP 0.093

> **Policy Insight:** Subsidizing low-income households has far greater impact on reducing catastrophic expenditure than disease-specific interventions.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Collection | Python, World Bank API, WHO GHO API, Kaggle |
| Data Processing | Pandas, NumPy, SQL |
| Machine Learning | Scikit-learn, XGBoost, SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
# Clone repo
git clone https://github.com/Asif5588-M/catastrophic-health-expenditure-pakistan.git
cd catastrophic-health-expenditure-pakistan

# Create environment
conda create -n chep-env python=3.11 -y
conda activate chep-env

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/streamlit_app.py
```

---

## 📈 Key Findings

- Pakistan has **never met the WHO 40% OOP threshold** in 24 years of data
- **73% of simulated cases** fall in the "catastrophic crisis" severity category
- Low-income households (< $300/month) face **near 100% catastrophic rate**
- Government health spending remains critically low at **~1% of CHE**
- Rising GDP per capita has **not proportionally reduced** OOP burden

---

## 👨‍💻 Author

**Asif Nawaz**
- 🏥 Medical Technician | PMAS Arid Agriculture University
- 🎓 MPhil Economics (Health Economics focus)
- 📄 Published Researcher — HEC Y-Category Journal
- 🔗 [Upwork Profile](https://www.upwork.com/freelancers/~016fa7751e0b328410)
- 🌐 [Live Dashboard](https://asifnawaz-pakistan-health.streamlit.app)

---

## 📄 Citation

If you use this work in your research, please cite:

```
Nawaz, A. (2025). Predicting Catastrophic Health Expenditure in Pakistan
using Machine Learning. PMAS Arid Agriculture University.
GitHub: https://github.com/Asif5588-M/catastrophic-health-expenditure-pakistan
```

---

*This project supports WHO Sustainable Development Goal 3.8 — 
Achieve universal health coverage, including financial risk protection.*