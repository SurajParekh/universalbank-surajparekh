import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Bank Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏦 Bank Personal Loan Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive Analytics & Propensity Prediction System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

# Column definitions
COLUMN_INFO = {
    'ID': 'Customer ID (will be dropped)',
    'Age': 'Customer age in completed years',
    'Experience': 'Years of professional experience',
    'Income': 'Annual income ($000)',
    'ZIP Code': 'Home Address ZIP code (will be dropped)',
    'Family': 'Family size of the customer',
    'CCAvg': 'Avg. spending on credit cards per month ($000)',
    'Education': 'Education Level (1: Undergrad, 2: Graduate, 3: Advanced)',
    'Mortgage': 'Value of house mortgage ($000)',
    'Personal Loan': 'TARGET: Accepted personal loan? (1=Yes, 0=No)',
    'Securities Account': 'Has securities account? (1=Yes, 0=No)',
    'CD Account': 'Has CD account? (1=Yes, 0=No)',
    'Online': 'Uses internet banking? (1=Yes, 0=No)',
    'CreditCard': 'Has bank credit card? (1=Yes, 0=No)'
}

@st.cache_data
def load_data(file):
    """Load and cache the dataset"""
    df = pd.read_csv(file)
    return df

def preprocess_data(df):
    """Preprocess data for modeling"""
    df_processed = df.copy()
    
    # Drop ID and ZIP Code
    cols_to_drop = ['ID', 'ZIP Code']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    # Handle negative experience (if any)
    if 'Experience' in df_processed.columns:
        df_processed['Experience'] = df_processed['Experience'].apply(lambda x: max(0, x))
    
    return df_processed

def get_features_target(df):
    """Split features and target"""
    X = df.drop('Personal Loan', axis=1)
    y = df['Personal Loan']
    return X, y

# ============================================
# DESCRIPTIVE ANALYTICS
# ============================================
def descriptive_analytics(df):
    """Perform descriptive analytics"""
    st.header("📊 Descriptive Analytics")
    st.markdown("*Understanding what happened - Summary statistics and distributions*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        loan_rate = df['Personal Loan'].mean() * 100
        st.metric("Loan Acceptance Rate", f"{loan_rate:.1f}%")
    with col3:
        avg_income = df['Income'].mean()
        st.metric("Avg. Income ($000)", f"${avg_income:.1f}K")
    with col4:
        avg_age = df['Age'].mean()
        st.metric("Avg. Age", f"{avg_age:.1f} years")
    
    st.markdown("---")
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_df = df[numeric_cols].describe().T
    summary_df['missing'] = df[numeric_cols].isnull().sum()
    summary_df['unique'] = df[numeric_cols].nunique()
    
    st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Distribution Plots
    st.subheader("📊 Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution
        fig_age = px.histogram(
            df, x='Age', nbins=30,
            title='Age Distribution',
            color_discrete_sequence=['#1E3A8A']
        )
        fig_age.update_layout(showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Income Distribution
        fig_income = px.histogram(
            df, x='Income', nbins=30,
            title='Income Distribution ($000)',
            color_discrete_sequence=['#059669']
        )
        fig_income.update_layout(showlegend=False)
        st.plotly_chart(fig_income, use_container_width=True)
    
    with col2:
        # Education Distribution
        edu_labels = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'}
        df_edu = df.copy()
        df_edu['Education_Label'] = df_edu['Education'].map(edu_labels)
        fig_edu = px.pie(
            df_edu, names='Education_Label',
            title='Education Level Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_edu, use_container_width=True)
        
        # Family Size Distribution
        fig_family = px.histogram(
            df, x='Family', nbins=4,
            title='Family Size Distribution',
            color_discrete_sequence=['#DC2626']
        )
        fig_family.update_layout(showlegend=False)
        st.plotly_chart(fig_family, use_container_width=True)
    
    st.markdown("---")
    
    # Target Variable Analysis
    st.subheader("🎯 Target Variable: Personal Loan Acceptance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loan_counts = df['Personal Loan'].value_counts()
        fig_target = px.pie(
            values=loan_counts.values,
            names=['No (0)', 'Yes (1)'],
            title='Personal Loan Acceptance Distribution',
            color_discrete_sequence=['#EF4444', '#10B981']
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        st.markdown("### Key Observations")
        no_loan = loan_counts.get(0, 0)
        yes_loan = loan_counts.get(1, 0)
        st.markdown(f"""
        - **Customers who declined loan:** {no_loan:,} ({no_loan/len(df)*100:.1f}%)
        - **Customers who accepted loan:** {yes_loan:,} ({yes_loan/len(df)*100:.1f}%)
        - **Class Imbalance Ratio:** {no_loan/yes_loan:.1f}:1
        
        ⚠️ *Note: This is an imbalanced dataset. The model evaluation should consider this.*
        """)

# ============================================
# DIAGNOSTIC ANALYTICS
# ============================================
def diagnostic_analytics(df):
    """Perform diagnostic analytics"""
    st.header("🔍 Diagnostic Analytics")
    st.markdown("*Understanding why it happened - Correlations and patterns*")
    
    df_processed = preprocess_data(df)
    
    # Correlation Heatmap
    st.subheader("🔗 Correlation Analysis")
    
    corr_matrix = df_processed.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Correlation with Target
    st.subheader("🎯 Correlation with Personal Loan (Target)")
    
    target_corr = corr_matrix['Personal Loan'].drop('Personal Loan').sort_values(ascending=True)
    
    fig_target_corr = px.bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        title='Feature Correlation with Personal Loan Acceptance',
        color=target_corr.values,
        color_continuous_scale='RdYlGn'
    )
    fig_target_corr.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_target_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Segmentation Analysis
    st.subheader("📊 Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Loan Acceptance
        fig_income_loan = px.box(
            df, x='Personal Loan', y='Income',
            title='Income Distribution by Loan Acceptance',
            color='Personal Loan',
            color_discrete_map={0: '#EF4444', 1: '#10B981'}
        )
        fig_income_loan.update_xaxes(tickvals=[0, 1], ticktext=['No', 'Yes'])
        st.plotly_chart(fig_income_loan, use_container_width=True)
        
        # Education vs Loan Acceptance
        edu_loan = df.groupby(['Education', 'Personal Loan']).size().unstack(fill_value=0)
        edu_loan_pct = edu_loan.div(edu_loan.sum(axis=1), axis=0) * 100
        
        fig_edu_loan = px.bar(
            edu_loan_pct.reset_index(),
            x='Education',
            y=[0, 1],
            title='Loan Acceptance Rate by Education Level',
            barmode='stack',
            color_discrete_map={0: '#EF4444', 1: '#10B981'}
        )
        fig_edu_loan.update_xaxes(tickvals=[1, 2, 3], ticktext=['Undergrad', 'Graduate', 'Advanced'])
        st.plotly_chart(fig_edu_loan, use_container_width=True)
    
    with col2:
        # CCAvg vs Loan Acceptance
        fig_cc_loan = px.box(
            df, x='Personal Loan', y='CCAvg',
            title='Credit Card Spending by Loan Acceptance',
            color='Personal Loan',
            color_discrete_map={0: '#EF4444', 1: '#10B981'}
        )
        fig_cc_loan.update_xaxes(tickvals=[0, 1], ticktext=['No', 'Yes'])
        st.plotly_chart(fig_cc_loan, use_container_width=True)
        
        # CD Account vs Loan Acceptance
        cd_loan = df.groupby(['CD Account', 'Personal Loan']).size().unstack(fill_value=0)
        cd_loan_pct = cd_loan.div(cd_loan.sum(axis=1), axis=0) * 100
        
        fig_cd_loan = px.bar(
            cd_loan_pct.reset_index(),
            x='CD Account',
            y=[0, 1],
            title='Loan Acceptance Rate by CD Account Status',
            barmode='stack',
            color_discrete_map={0: '#EF4444', 1: '#10B981'}
        )
        fig_cd_loan.update_xaxes(tickvals=[0, 1], ticktext=['No CD Account', 'Has CD Account'])
        st.plotly_chart(fig_cd_loan, use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("💡 Key Diagnostic Insights")
    
    # Calculate insights
    avg_income_yes = df[df['Personal Loan'] == 1]['Income'].mean()
    avg_income_no = df[df['Personal Loan'] == 0]['Income'].mean()
    
    avg_cc_yes = df[df['Personal Loan'] == 1]['CCAvg'].mean()
    avg_cc_no = df[df['Personal Loan'] == 0]['CCAvg'].mean()
    
    cd_yes_rate = df[df['CD Account'] == 1]['Personal Loan'].mean() * 100
    cd_no_rate = df[df['CD Account'] == 0]['Personal Loan'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        ### 💰 Income Impact
        - Loan acceptors avg income: **${avg_income_yes:.1f}K**
        - Non-acceptors avg income: **${avg_income_no:.1f}K**
        - **{avg_income_yes/avg_income_no:.1f}x higher** income for acceptors
        """)
    
    with col2:
        st.markdown(f"""
        ### 💳 Credit Card Spending
        - Loan acceptors avg CC spend: **${avg_cc_yes:.2f}K/month**
        - Non-acceptors avg CC spend: **${avg_cc_no:.2f}K/month**
        - **{avg_cc_yes/avg_cc_no:.1f}x higher** spending for acceptors
        """)
    
    with col3:
        st.markdown(f"""
        ### 🏦 CD Account Effect
        - With CD Account: **{cd_yes_rate:.1f}%** acceptance
        - Without CD Account: **{cd_no_rate:.1f}%** acceptance
        - CD holders are **{cd_yes_rate/cd_no_rate:.1f}x more likely** to accept
        """)

# ============================================
# PREDICTIVE ANALYTICS
# ============================================
def predictive_analytics(df):
    """Perform predictive analytics with ML models"""
    st.header("🤖 Predictive Analytics")
    st.markdown("*Predicting what will happen - Machine Learning Models*")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    X, y = get_features_target(df_processed)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.subheader("⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌳 Decision Tree")
        dt_max_depth = st.slider("Max Depth (DT)", 1, 20, 5, key='dt_depth')
        dt_min_samples = st.slider("Min Samples Split (DT)", 2, 20, 2, key='dt_samples')
    
    with col2:
        st.markdown("### 🌲 Random Forest")
        rf_n_estimators = st.slider("N Estimators (RF)", 10, 200, 100, key='rf_est')
        rf_max_depth = st.slider("Max Depth (RF)", 1, 20, 10, key='rf_depth')
    
    with col3:
        st.markdown("### 🚀 Gradient Boosting")
        gb_n_estimators = st.slider("N Estimators (GB)", 10, 200, 100, key='gb_est')
        gb_learning_rate = st.slider("Learning Rate (GB)", 0.01, 0.5, 0.1, key='gb_lr')
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... Please wait."):
            # Initialize models
            models = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=dt_max_depth,
                    min_samples_split=dt_min_samples,
                    random_state=42
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=gb_n_estimators,
                    learning_rate=gb_learning_rate,
                    random_state=42
                )
            }
            
            results = {}
            
            for name, model in models.items():
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                # Metrics
                results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_prob),
                    'cv_scores': cross_val_score(model, X_train_scaled, y_train, cv=5)
                }
            
            # Store in session state
            st.session_state['results'] = results
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
            st.session_state['scaler'] = scaler
        
        st.success("✅ Models trained successfully!")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']
        
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics comparison table
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'Precision': [results[m]['precision'] for m in results],
            'Recall': [results[m]['recall'] for m in results],
            'F1 Score': [results[m]['f1'] for m in results],
            'ROC AUC': [results[m]['roc_auc'] for m in results],
            'CV Mean': [results[m]['cv_scores'].mean() for m in results],
            'CV Std': [results[m]['cv_scores'].std() for m in results]
        })
        
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'ROC AUC': '{:.4f}',
                'CV Mean': '{:.4f}',
                'CV Std': '{:.4f}'
            }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], color='lightgreen'),
            use_container_width=True
        )
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            fig_metrics = px.bar(
                metrics_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
                x='variable', y='value', color='Model',
                barmode='group',
                title='Model Performance Metrics Comparison',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_metrics.update_layout(xaxis_title='Metric', yaxis_title='Score')
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # ROC Curves
            fig_roc = go.Figure()
            
            colors = ['#1E3A8A', '#059669', '#DC2626']
            for i, (name, res) in enumerate(results.items()):
                fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={res['roc_auc']:.3f})",
                    line=dict(color=colors[i], width=2)
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_roc.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrices
        st.subheader("📉 Confusion Matrices")
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                cm = confusion_matrix(y_test, res['y_pred'])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Loan', 'Loan'],
                    y=['No Loan', 'Loan'],
                    title=f'{name}',
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("📊 Feature Importance")
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                importance = res['model'].feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(
                    feat_imp_df,
                    x='Importance', y='Feature',
                    orientation='h',
                    title=f'{name}',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_imp.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
        
        # Best Model Summary
        st.markdown("---")
        st.subheader("🏆 Best Model Summary")
        
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        best_result = results[best_model_name]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            ### Winner: **{best_model_name}**
            
            | Metric | Score |
            |--------|-------|
            | Accuracy | {best_result['accuracy']:.4f} |
            | Precision | {best_result['precision']:.4f} |
            | Recall | {best_result['recall']:.4f} |
            | F1 Score | {best_result['f1']:.4f} |
            | ROC AUC | {best_result['roc_auc']:.4f} |
            | CV Mean ± Std | {best_result['cv_scores'].mean():.4f} ± {best_result['cv_scores'].std():.4f} |
            """)
        
        with col2:
            st.markdown(f"""
            ### 📝 Classification Report - {best_model_name}
            ```
            {classification_report(y_test, best_result['y_pred'], target_names=['No Loan', 'Loan'])}
            ```
            """)

# ============================================
# PRESCRIPTIVE ANALYTICS
# ============================================
def prescriptive_analytics(df):
    """Perform prescriptive analytics"""
    st.header("💡 Prescriptive Analytics")
    st.markdown("*Recommending what should be done - Actionable insights*")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Please run Predictive Analytics first to train the models.")
        return
    
    results = st.session_state['results']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    
    # Use best model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    st.success(f"Using **{best_model_name}** for recommendations (Best ROC AUC)")
    
    st.markdown("---")
    
    # Customer Segmentation Recommendations
    st.subheader("🎯 Customer Targeting Recommendations")
    
    df_processed = preprocess_data(df)
    X, y = get_features_target(df_processed)
    
    # Get predictions for all customers
    X_scaled = scaler.transform(X)
    probabilities = best_model.predict_proba(X_scaled)[:, 1]
    
    df_results = df.copy()
    df_results['Loan_Probability'] = probabilities
    df_results['Risk_Segment'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Segment Analysis
    col1, col2 = st.columns(2)
    
with col1:
    segment_counts = df_results['Risk_Segment'].value_counts()
    
    # Define colors for segments (Low=Red, Medium=Orange, High=Light Green, Very High=Green)
    segment_colors = {
        'Low': '#EF4444',
        'Medium': '#F59E0B', 
        'High': '#10B981',
        'Very High': '#059669'
    }
    
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title='Customer Segments by Loan Propensity',
        color=segment_counts.index,
        color_discrete_map=segment_colors
    )
    st.plotly_chart(fig_segments, use_container_width=True)    
    with col2:
        fig_prob_dist = px.histogram(
            df_results, x='Loan_Probability', nbins=50,
            title='Distribution of Loan Acceptance Probability',
            color_discrete_sequence=['#1E3A8A']
        )
        fig_prob_dist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_prob_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Segment Profiles
    st.subheader("📊 Segment Profiles")
    
    segment_profile = df_results.groupby('Risk_Segment').agg({
        'Age': 'mean',
        'Income': 'mean',
        'CCAvg': 'mean',
        'Education': 'mean',
        'Mortgage': 'mean',
        'CD Account': 'mean',
        'Loan_Probability': ['mean', 'count']
    }).round(2)
    
    segment_profile.columns = ['Avg Age', 'Avg Income ($K)', 'Avg CC Spend ($K)', 
                                'Avg Education', 'Avg Mortgage ($K)', 'CD Account %',
                                'Avg Probability', 'Customer Count']
    segment_profile['CD Account %'] = (segment_profile['CD Account %'] * 100).round(1)
    
    st.dataframe(segment_profile.style.background_gradient(cmap='RdYlGn', subset=['Avg Probability']), 
                 use_container_width=True)
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.subheader("📋 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 High Priority Targets (Very High Segment)
        
        **Profile:**
        - High income customers
        - Higher credit card spending
        - Often have CD accounts
        - Higher education levels
        
        **Recommended Actions:**
        1. ✅ **Personalized outreach** via relationship managers
        2. ✅ **Premium loan packages** with competitive rates
        3. ✅ **Cross-sell** with investment products
        4. ✅ **Priority processing** for loan applications
        
        ---
        
        ### 📈 Growth Opportunities (High Segment)
        
        **Recommended Actions:**
        1. 📧 **Email campaigns** with loan benefits
        2. 💰 **Promotional interest rates**
        3. 📱 **Mobile app notifications**
        4. 🎁 **Bundled offers** with existing products
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Nurture Campaigns (Medium Segment)
        
        **Recommended Actions:**
        1. 📚 **Educational content** about loan benefits
        2. 💳 **Increase engagement** via credit card offers
        3. 🏦 **CD account promotions** (strong predictor)
        4. ⏰ **Retarget** after 3-6 months
        
        ---
        
        ### 📊 Long-term Strategy (Low Segment)
        
        **Recommended Actions:**
        1. 🎓 **Financial literacy programs**
        2. 💼 **Entry-level products** to build relationship
        3. 📈 **Monitor for life changes** (income increase, etc.)
        4. 🔄 **Re-score quarterly**
        """)
    
    st.markdown("---")
    
    # Top Prospects List
    st.subheader("🏆 Top 20 Prospects for Immediate Outreach")
    
    top_prospects = df_results.nlargest(20, 'Loan_Probability')[
        ['ID', 'Age', 'Income', 'Education', 'CCAvg', 'CD Account', 'Loan_Probability', 'Risk_Segment']
    ]
    top_prospects['Education'] = top_prospects['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'})
    top_prospects['CD Account'] = top_prospects['CD Account'].map({0: 'No', 1: 'Yes'})
    top_prospects['Loan_Probability'] = (top_prospects['Loan_Probability'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(top_prospects, use_container_width=True)
    
    # Download button for prospects
    csv = top_prospects.to_csv(index=False)
    st.download_button(
        label="📥 Download Top Prospects List",
        data=csv,
        file_name="top_prospects.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # ROI Estimation
    st.subheader("💰 Estimated Campaign ROI")
    
    col1, col2, col3 = st.columns(3)
    
    very_high_count = len(df_results[df_results['Risk_Segment'] == 'Very High'])
    high_count = len(df_results[df_results['Risk_Segment'] == 'High'])
    
    with col1:
        st.metric("Very High Propensity Customers", f"{very_high_count:,}")
        st.caption("Expected conversion: 70-90%")
    
    with col2:
        st.metric("High Propensity Customers", f"{high_count:,}")
        st.caption("Expected conversion: 50-70%")
    
    with col3:
        total_target = very_high_count + high_count
        st.metric("Total Priority Targets", f"{total_target:,}")
        st.caption(f"{total_target/len(df)*100:.1f}% of customer base")

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Display data info in sidebar
        st.sidebar.success(f"✅ Loaded {len(df):,} rows")
        st.sidebar.markdown("---")
        
        # Data preview
        st.sidebar.subheader("📋 Column Info")
        for col in df.columns:
            if col in COLUMN_INFO:
                st.sidebar.caption(f"**{col}**: {COLUMN_INFO[col]}")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Overview",
            "📊 Descriptive",
            "🔍 Diagnostic", 
            "🤖 Predictive",
            "💡 Prescriptive"
        ])
        
        with tab1:
            st.header("📋 Data Overview")
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Data Shape")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", f"{df.shape[1]}")
            
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.notnull().sum().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab2:
            descriptive_analytics(df)
        
        with tab3:
            diagnostic_analytics(df)
        
        with tab4:
            predictive_analytics(df)
        
        with tab5:
            prescriptive_analytics(df)
    
    else:
        # Welcome screen
        st.info("👈 Please upload your CSV file using the sidebar to get started.")
        
        st.markdown("""
        ## 🚀 How to Use This App
        
        1. **Upload** your bank customer data CSV file
        2. **Explore** the data in the Overview tab
        3. **Analyze** using the four analytics tabs:
           - 📊 **Descriptive**: What happened?
           - 🔍 **Diagnostic**: Why did it happen?
           - 🤖 **Predictive**: What will happen?
           - 💡 **Prescriptive**: What should we do?
        
        ## 📊 Expected Data Format
        
        Your CSV should contain these columns:
        - `ID`, `Age`, `Experience`, `Income`, `ZIP Code`
        - `Family`, `CCAvg`, `Education`, `Mortgage`
        - `Personal Loan` (Target), `Securities Account`
        - `CD Account`, `Online`, `CreditCard`
        """)

if __name__ == "__main__":
    main()
