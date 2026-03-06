# 🏦 Bank Personal Loan Analytics

A comprehensive Streamlit application for analyzing bank customer data and predicting personal loan acceptance propensity.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This application performs **four types of analytics** on bank customer data:

| Analytics Type | Description |
|----------------|-------------|
| **Descriptive** | Summary statistics, distributions, and visualizations |
| **Diagnostic** | Correlation analysis, segmentation, and pattern discovery |
| **Predictive** | ML models (Decision Tree, Random Forest, Gradient Boosting) |
| **Prescriptive** | Customer targeting recommendations and actionable insights |

## 🚀 Features

- ✅ Interactive data upload and exploration
- ✅ Comprehensive statistical analysis
- ✅ Beautiful visualizations with Plotly
- ✅ Three classification algorithms with hyperparameter tuning
- ✅ Model comparison and evaluation metrics
- ✅ ROC curves and confusion matrices
- ✅ Feature importance analysis
- ✅ Customer segmentation and targeting recommendations
- ✅ Downloadable prospect lists

## 📊 Dataset

The application expects a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| ID | Customer ID |
| Age | Customer age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| ZIP Code | Home address ZIP code |
| Family | Family size |
| CCAvg | Avg. credit card spending per month ($000) |
| Education | 1: Undergrad, 2: Graduate, 3: Advanced |
| Mortgage | House mortgage value ($000) |
| Personal Loan | **Target**: Accepted loan? (1=Yes, 0=No) |
| Securities Account | Has securities account? (1/0) |
| CD Account | Has CD account? (1/0) |
| Online | Uses internet banking? (1/0) |
| CreditCard | Has bank credit card? (1/0) |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bank-loan-analytics.git
cd bank-loan-analytics
