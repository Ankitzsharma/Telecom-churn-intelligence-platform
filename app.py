import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data
from src.preprocessing import clean_data, encode_data
from src.feature_engineering import create_features
from src.business_logic import simulate_financial_impact

# Page Config
st.set_page_config(
    page_title="Telecom Customer Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        /* Global Settings */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E2E8F0;
        }
        
        /* Background & Main Container */
        .stApp {
            background-color: #0F172A;
        }
        
        /* Typography */
        h1 {
            font-size: 34px !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            margin-bottom: 24px !important;
        }
        
        h2 {
            font-size: 26px !important;
            font-weight: 600 !important;
            color: #F1F5F9 !important;
            margin-top: 32px !important;
        }
        
        h3 {
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #E2E8F0 !important;
        }
        
        /* KPI Cards */
        .kpi-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: transform 0.2s ease-in-out;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            border-color: #0EA5E9;
            box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.1);
        }
        
        .kpi-value {
            font-size: 40px;
            font-weight: 700;
            color: #F8FAFC;
            margin: 8px 0;
        }
        
        .kpi-label {
            font-size: 14px;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        
        /* Insight Boxes */
        .insight-box {
            background-color: #1E293B;
            border-left: 4px solid #0EA5E9;
            padding: 16px;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
            color: #CBD5E1;
        }
        
        /* Custom Buttons (Navbar & Form) */
        .stButton>button {
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        /* Secondary Button (Inactive Nav) */
        .stButton>button[kind="secondary"] {
            background-color: transparent;
            color: #94A3B8;
            border: 1px solid #334155;
        }
        
        .stButton>button[kind="secondary"]:hover {
            background-color: rgba(30, 41, 59, 0.5);
            color: #E2E8F0;
            border-color: #475569;
        }

        /* Primary Button (Active Nav & Actions) */
        .stButton>button[kind="primary"] {
            background-color: #0EA5E9;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.3);
        }
        
        .stButton>button[kind="primary"]:hover {
            background-color: #0284C7;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #1E293B;
            border-radius: 8px;
            color: #E2E8F0 !important;
        }
        
        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            color: #F8FAFC;
        }
        
        [data-testid="stMetricLabel"] {
            color: #94A3B8;
        }
        
        /* Hide default header and sidebar */
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Executive Overview'

def set_page(page_name):
    st.session_state.page = page_name

# Top Navbar
st.markdown("""
    <div style="background-color: #1E293B; padding: 15px 20px; border-radius: 12px; margin-bottom: 25px; border: 1px solid #334155; display: flex; align-items: center; justify-content: space-between;">
        <div style="font-weight: 700; font-size: 20px; color: #F8FAFC; display: flex; align-items: center;">
            <span style="background: linear-gradient(45deg, #0EA5E9, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-right: 10px;">📡</span>
            Telecom Analytics <span style="font-size: 12px; background: #334155; color: #94A3B8; padding: 2px 8px; border-radius: 12px; margin-left: 10px;">PRO v2.0</span>
        </div>
        <div style="font-size: 12px; color: #64748B;">
            Enterprise Customer Intelligence Platform
        </div>
    </div>
""", unsafe_allow_html=True)

# Navigation Buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)
nav_items = [
    "Executive Overview", 
    "Customer Segmentation", 
    "Churn Prediction", 
    "Financial Simulator", 
    "Model Insights", 
    "System Architecture"
]

for i, (col, item) in enumerate(zip([col1, col2, col3, col4, col5, col6], nav_items)):
    with col:
        # Highlight active page using primary button type
        btn_type = "primary" if st.session_state.page == item else "secondary"
        if st.button(item, key=f"nav_{i}", type=btn_type, use_container_width=True):
            set_page(item)
            st.rerun()

# Load Data and Models
@st.cache_data
def load_dataset():
    filepath = "Data/Telco-Customer-Churn-Dataset.csv"
    if not os.path.exists(filepath):
        st.error("Dataset not found! Please check path.")
        return None
    df = load_data(filepath)
    df = clean_data(df)
    df = create_features(df)
    return df

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('models/scaler.pkl')
        kmeans = joblib.load('models/kmeans.pkl')
        cluster_names = joblib.load('models/cluster_names.pkl')
        best_model = joblib.load('models/best_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return scaler, kmeans, cluster_names, best_model, feature_names
    except FileNotFoundError:
        st.error("Models not found! Please run 'main.py' first to train and save models.")
        return None, None, None, None, None

# Load resources
df = load_dataset()
scaler, kmeans, cluster_names, best_model, feature_names = load_models()

# Pre-calculation for global use
if df is not None and best_model is not None:
    if 'Cluster_Label' not in df.columns:
        features_for_clustering = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 
            'EngagementScore', 'RevenueIntensity', 'EstimatedCLV'
        ]
        X_cluster = df[features_for_clustering].copy()
        X_scaled = scaler.transform(X_cluster)
        clusters = kmeans.predict(X_scaled)
        df['Cluster'] = clusters
        df['Cluster_Label'] = df['Cluster'].map(cluster_names)

    # Page Routing
    page = st.session_state.page

    # --- SECTION 1: EXECUTIVE OVERVIEW ---
    if page == "Executive Overview":
        st.title("Executive Overview")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>Real-time performance metrics and high-level churn indicators.</p>", unsafe_allow_html=True)
        
        # Key Performance Indicators
        total_customers = len(df)
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        avg_revenue = df['MonthlyCharges'].mean()
        high_risk_count = (df['Cluster_Label'] == 'High Revenue At Risk').sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Total Customers", f"{total_customers:,}", ""),
            ("Churn Rate", f"{churn_rate:.1f}%", "⚠️ Critical"),
            ("Avg Monthly Revenue", f"${avg_revenue:.2f}", ""),
            ("High Risk Customers", f"{high_risk_count:,}", "🚨 Action Needed")
        ]
        
        for col, (label, value, tag) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                        <div style="color: #EF4444; font-size: 12px; font-weight: 600; min-height: 18px;">{tag}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown("### Strategic Analysis")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### Overall Churn Distribution")
            churn_counts = df['Churn'].value_counts().reset_index()
            churn_counts.columns = ['Churn', 'Count']
            
            # FIXED: Use px.pie with hole instead of px.donut (which doesn't exist)
            fig_churn = px.pie(churn_counts, values='Count', names='Churn', 
                               color='Churn',
                               color_discrete_map={'No':'#10B981', 'Yes':'#EF4444'},
                               hole=0.6)
            fig_churn.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0),
                annotations=[dict(text=f"{churn_rate:.1f}%<br>Churn", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_churn, use_container_width=True)
            st.markdown("<div class='insight-box'>💡 <b>Insight:</b> 26.5% of the customer base has churned, exceeding the industry average of 21%.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_chart2:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### Revenue Impact by Contract")
            rev_contract = df.groupby('Contract')['MonthlyCharges'].sum().reset_index()
            
            fig_rev = px.bar(rev_contract, x='Contract', y='MonthlyCharges', 
                             color='Contract',
                             color_discrete_sequence=['#3B82F6', '#6366F1', '#8B5CF6'])
            fig_rev.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                showlegend=False,
                margin=dict(t=20, b=0, l=0, r=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#334155')
            )
            fig_rev.update_traces(marker_line_width=0, opacity=0.9, width=0.5)
            st.plotly_chart(fig_rev, use_container_width=True)
            st.markdown("<div class='insight-box'>💡 <b>Insight:</b> Month-to-month contracts generate high revenue but contribute to 88% of total churn volume.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- SECTION 2: CUSTOMER SEGMENTATION ---
    elif page == "Customer Segmentation":
        st.title("Customer Segmentation Intelligence")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>AI-driven clustering of customer base into behavioral personas.</p>", unsafe_allow_html=True)
        
        # Cluster Stats
        cluster_stats = df.groupby('Cluster_Label').agg(
            Count=('tenure', 'count'),
            Avg_Revenue=('MonthlyCharges', 'mean'),
            Avg_Tenure=('tenure', 'mean'),
            Churn_Rate=('Churn', lambda x: (x=='Yes').mean())
        ).reset_index()
        cluster_stats.columns = ['Segment', 'Count', 'Avg Revenue', 'Avg Tenure', 'Churn Rate']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### Segment Distribution")
            fig_seg = px.bar(cluster_stats, x='Segment', y='Count', color='Segment', 
                             text='Count', 
                             color_discrete_sequence=px.colors.qualitative.Prism)
            fig_seg.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#334155')
            )
            fig_seg.update_traces(marker_line_width=0, textposition='outside')
            st.plotly_chart(fig_seg, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("#### Segment Profiles")
            st.dataframe(cluster_stats.style.format({
                'Avg Revenue': '${:.2f}',
                'Avg Tenure': '{:.1f} mo',
                'Churn Rate': '{:.1%}'
            }), height=300, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("### Behavioral Deep Dive")
        st.markdown("<div class='insight-box'>💡 <b>Executive Insight:</b> 'High Revenue At Risk' customers show high spending power but exhibit declining engagement scores, indicating potential silent attrition.</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            fig_scat = px.scatter(df, x='tenure', y='MonthlyCharges', color='Cluster_Label',
                                  title="Tenure vs Revenue Mapping", opacity=0.7,
                                  color_discrete_sequence=px.colors.qualitative.Prism)
            fig_scat.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                xaxis=dict(showgrid=True, gridcolor='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#334155')
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            fig_box = px.box(df, x='Cluster_Label', y='EngagementScore', color='Cluster_Label',
                             title="Engagement Score Variance",
                             color_discrete_sequence=px.colors.qualitative.Prism)
            fig_box.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#334155')
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- SECTION 3: CHURN PREDICTION ---
    elif page == "Churn Prediction":
        st.title("Individual Customer Prediction")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>Predict churn probability and identify risk factors for a specific customer profile.</p>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.markdown("#### 1. Demographics")
            col1, col2, col3, col4 = st.columns(4)
            with col1: gender = st.selectbox("Gender", ["Male", "Female"])
            with col2: senior = st.selectbox("Senior Citizen", [0, 1])
            with col3: partner = st.selectbox("Partner", ["Yes", "No"])
            with col4: dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            st.markdown("#### 2. Services")
            col1, col2, col3, col4 = st.columns(4)
            with col1: tenure = st.slider("Tenure (Months)", 0, 72, 12)
            with col2: phone = st.selectbox("Phone Service", ["Yes", "No"])
            with col3: internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            with col4: multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            with col2: backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            with col3: device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            with col4: tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            
            col1, col2 = st.columns(2)
            with col1: tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            with col2: movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            st.markdown("#### 3. Billing & Contract")
            col1, col2, col3 = st.columns(3)
            with col1: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            with col2: paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            with col3: payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            col1, col2 = st.columns(2)
            with col1: monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
            with col2: total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

            st.markdown("---")
            submit = st.form_submit_button("Run Prediction Model")
            
        if submit:
            # Create DataFrame from input
            input_data = {
                'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner], 'Dependents': [dependents],
                'tenure': [tenure], 'PhoneService': [phone], 'MultipleLines': [multiple], 
                'InternetService': [internet], 'OnlineSecurity': [security], 'OnlineBackup': [backup],
                'DeviceProtection': [device], 'TechSupport': [tech], 'StreamingTV': [tv], 
                'StreamingMovies': [movies], 'Contract': [contract], 'PaperlessBilling': [paperless],
                'PaymentMethod': [payment], 'MonthlyCharges': [monthly], 'TotalCharges': [total]
            }
            input_df = pd.DataFrame(input_data)
            
            # Processing pipeline
            input_df = create_features(input_df)
            
            # Segmentation
            features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges', 'EngagementScore', 'RevenueIntensity', 'EstimatedCLV']
            X_cluster = input_df[features_for_clustering].copy()
            X_scaled = scaler.transform(X_cluster)
            cluster_pred = kmeans.predict(X_scaled)[0]
            input_df['Cluster'] = cluster_pred
            cluster_label = cluster_names[cluster_pred]
            input_df['Cluster_Label'] = cluster_label
            
            # Encoding & Prediction
            input_encoded = pd.get_dummies(input_df)
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_final = input_encoded[feature_names]
            
            prob = best_model.predict_proba(input_final)[0][1]
            
            # Result Display
            st.markdown("### Prediction Results")
            
            risk_color = "#EF4444" if prob > 0.5 else "#10B981"
            risk_label = "High Risk" if prob > 0.5 else "Safe"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Churn Probability</div>
                        <div class="kpi-value">{prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="kpi-card" style="border-color: {risk_color};">
                        <div class="kpi-label">Risk Level</div>
                        <div class="kpi-value" style="color: {risk_color};">{risk_label}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Customer Segment</div>
                        <div class="kpi-value" style="font-size: 24px; margin-top: 18px;">{cluster_label}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown("#### 💡 Strategic Recommendation")
            if prob > 0.7:
                st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; padding: 16px; border-radius: 4px;">
                    <h4 style="color: #EF4444; margin:0;">🚨 Critical Retention Action Required</h4>
                    <p style="margin-top: 8px;">Customer is highly likely to churn. Recommend immediate intervention:</p>
                    <ul>
                        <li>Offer 20% discount on 1-year contract renewal.</li>
                        <li>Schedule priority support call to address technical issues.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif prob > 0.4:
                st.markdown("""
                <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; padding: 16px; border-radius: 4px;">
                    <h4 style="color: #F59E0B; margin:0;">⚠️ Proactive Engagement Suggested</h4>
                    <p style="margin-top: 8px;">Customer showing early signs of risk.</p>
                    <ul>
                        <li>Send educational content on service benefits.</li>
                        <li>Highlight bundle savings opportunities.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 16px; border-radius: 4px;">
                    <h4 style="color: #10B981; margin:0;">✅ Upsell Opportunity</h4>
                    <p style="margin-top: 8px;">Customer is loyal and stable.</p>
                    <ul>
                        <li>Recommend premium add-ons (Streaming, Security).</li>
                        <li>Offer referral bonuses.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # --- SECTION 4: FINANCIAL SIMULATOR ---
    elif page == "Financial Simulator":
        st.title("Financial Impact Simulator")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>Estimate the ROI of retention campaigns by targeting high-risk segments.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div style='background: #1E293B; padding: 24px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### Campaign Parameters")
            retention_rate = st.slider("Retention Success Rate (%)", 0, 100, 30) / 100
            cost_per_offer = st.number_input("Cost per Offer ($)", 0, 200, 10)
            
            if st.button("Run Simulation", use_container_width=True):
                with st.spinner("Calculating Impact..."):
                    df_encoded = encode_data(df)
                    for col in feature_names:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0
                    X_full = df_encoded[feature_names]
                    y_probs = best_model.predict_proba(X_full)[:, 1]
                    
                    threshold = np.quantile(y_probs, 0.8)
                    high_risk_mask = y_probs >= threshold
                    high_risk_customers = df[high_risk_mask]
                    
                    num_targeted = len(high_risk_customers)
                    potential_revenue_loss = high_risk_customers['MonthlyCharges'].sum()
                    saved_revenue = potential_revenue_loss * retention_rate
                    campaign_cost = num_targeted * cost_per_offer
                    roi = (saved_revenue - campaign_cost) / campaign_cost * 100 if campaign_cost > 0 else 0
                    
                    st.session_state['sim_results'] = {
                        'saved': saved_revenue,
                        'cost': campaign_cost,
                        'roi': roi,
                        'targeted': num_targeted
                    }
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            if 'sim_results' in st.session_state:
                res = st.session_state['sim_results']
                roi_color = "#10B981" if res['roi'] > 0 else "#EF4444"
                
                st.markdown(f"""
                <div style='background: #1E293B; padding: 32px; border-radius: 12px; text-align: center;'>
                    <h3 style='color: #94A3B8; margin-bottom: 8px;'>PROJECTED ROI</h3>
                    <div style='font-size: 64px; font-weight: 700; color: {roi_color}; text-shadow: 0 0 20px {roi_color}40;'>
                        {res['roi']:.1f}%
                    </div>
                    <p style='color: #CBD5E1; margin-top: 16px;'>
                        Based on targeting <b>{res['targeted']:,}</b> high-risk customers.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div style='background: #1E293B; padding: 20px; border-radius: 12px; margin-top: 16px; text-align: center;'>
                        <div style='color: #94A3B8; font-size: 14px;'>REVENUE SAVED</div>
                        <div style='color: #F8FAFC; font-size: 28px; font-weight: 600;'>${res['saved']:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style='background: #1E293B; padding: 20px; border-radius: 12px; margin-top: 16px; text-align: center;'>
                        <div style='color: #94A3B8; font-size: 14px;'>CAMPAIGN COST</div>
                        <div style='color: #F8FAFC; font-size: 28px; font-weight: 600;'>${res['cost']:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("👈 Adjust parameters and click 'Run Simulation' to see results.")

    # --- SECTION 5: MODEL INSIGHTS ---
    elif page == "Model Insights":
        st.title("Model Explainability")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>Understanding the key drivers behind churn predictions.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div style='background: #1E293B; padding: 20px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### Top Churn Drivers")
            
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_imp = feature_imp.sort_values('Importance', ascending=True).tail(10)
                
                fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                                 color='Importance',
                                 color_continuous_scale='Blues')
            elif hasattr(best_model, 'coef_'):
                coefs = best_model.coef_[0]
                feature_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
                feature_imp['AbsCoef'] = feature_imp['Coefficient'].abs()
                feature_imp = feature_imp.sort_values('AbsCoef', ascending=True).tail(10)
                
                fig_imp = px.bar(feature_imp, x='Coefficient', y='Feature', orientation='h',
                                 color='Coefficient',
                                 color_continuous_scale='RdBu_r')
            
            fig_imp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                xaxis=dict(showgrid=True, gridcolor='#334155'),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background: #1E293B; padding: 24px; border-radius: 12px; height: 100%;'>
                <h4>Interpretation Guide</h4>
                <p style='color: #CBD5E1; margin-top: 16px;'>
                    <b>Positive Values (Red/Blue):</b> Features that increase the likelihood of churn (e.g., Month-to-month contracts, Fiber optic service).
                </p>
                <p style='color: #CBD5E1; margin-top: 16px;'>
                    <b>Negative Values (Blue/Red):</b> Features that increase retention (e.g., Long tenure, Tech support subscription).
                </p>
                <div style='margin-top: 24px; padding: 16px; background: rgba(14, 165, 233, 0.1); border-radius: 8px;'>
                    <small style='color: #0EA5E9; font-weight: 600;'>DATA SCIENCE NOTE</small>
                    <p style='font-size: 14px; margin-top: 8px;'>
                        The model prioritizes behavioral features (Contract type, Payment method) over demographic features.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- SECTION 6: SYSTEM ARCHITECTURE ---
    elif page == "System Architecture":
        st.title("System Architecture")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 32px;'>End-to-end ML pipeline design and data flow.</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #1E293B; padding: 40px; border-radius: 12px; text-align: center;'>
            <div style='display: flex; justify-content: space-between; align-items: center; max-width: 900px; margin: 0 auto; flex-wrap: wrap;'>
                <div style='text-align: center; width: 150px;'>
                    <div style='font-size: 40px; margin-bottom: 10px;'>📂</div>
                    <h4 style='color: #F8FAFC;'>Data Source</h4>
                    <p style='font-size: 12px; color: #94A3B8;'>Telco CSV<br>Raw Ingestion</p>
                </div>
                <div style='color: #334155; font-size: 24px;'>➜</div>
                <div style='text-align: center; width: 150px;'>
                    <div style='font-size: 40px; margin-bottom: 10px;'>⚙️</div>
                    <h4 style='color: #F8FAFC;'>Preprocessing</h4>
                    <p style='font-size: 12px; color: #94A3B8;'>Cleaning<br>Feature Engineering</p>
                </div>
                <div style='color: #334155; font-size: 24px;'>➜</div>
                <div style='text-align: center; width: 150px;'>
                    <div style='font-size: 40px; margin-bottom: 10px;'>🤖</div>
                    <h4 style='color: #F8FAFC;'>Modeling</h4>
                    <p style='font-size: 12px; color: #94A3B8;'>XGBoost / LogReg<br>K-Means Clustering</p>
                </div>
                <div style='color: #334155; font-size: 24px;'>➜</div>
                <div style='text-align: center; width: 150px;'>
                    <div style='font-size: 40px; margin-bottom: 10px;'>🖥️</div>
                    <h4 style='color: #F8FAFC;'>Deployment</h4>
                    <p style='font-size: 12px; color: #94A3B8;'>Streamlit UI<br>Interactive Dashboard</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🛠 Tech Stack")
            st.markdown("""
            - **Language:** Python 3.10+
            - **Data Processing:** Pandas, NumPy
            - **Machine Learning:** Scikit-Learn, XGBoost
            - **Visualization:** Plotly Express
            - **App Framework:** Streamlit
            """)
        with col2:
            st.markdown("### 🔄 Pipeline Features")
            st.markdown("""
            - Automated schema validation
            - Leakage-free preprocessing pipeline
            - Modular feature engineering
            - Production-grade logging
            - Artifact management (Joblib)
            """)

else:
    st.warning("Please ensure data and models are loaded correctly.")
