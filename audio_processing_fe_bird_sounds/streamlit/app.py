import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# PAGE CONFIGURATION 

st.set_page_config(
    page_title="Bird Sound Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  API CONFIGURATION


def get_api_base_url():
    """Get API base URL from secrets, environment, or default"""
    try:
        return st.secrets["API_BASE_URL"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv("API_BASE_URL", "http://localhost:5000")

API_BASE_URL = get_api_base_url()


# CUSTOM STYLING

st.markdown("""
<style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Navigation Buttons */
    .nav-button {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #6366f1;
        padding: 15px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background: rgba(99, 102, 241, 0.2);
        border-left-color: #818cf8;
        transform: translateX(5px);
    }
    
    .nav-button.active {
        background: rgba(99, 102, 241, 0.3);
        border-left-color: #a5b4fc;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        padding: 28px;
        border-radius: 16px;
        border-left: 6px solid #6366f1;
        margin: 24px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Status Boxes */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #10b981;
        margin: 12px 0;
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #ef4444;
        margin: 12px 0;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin: 12px 0;
        font-weight: 500;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 40px 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        margin-bottom: 32px;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.2em;
        margin-bottom: 12px;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px 12px 0 0;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.95);
        color: #1e3c72;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: 700;
        color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)
# SESSION STATE INITIALIZATION

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'predict'
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'uploaded_files_count' not in st.session_state:
    st.session_state.uploaded_files_count = 0

# API HELPER FUNCTIONS

def get_system_health():
    """Get API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None

def get_metrics():
    """Get application metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/metrics", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def make_prediction(audio_file):
    """Send audio file for prediction"""
    try:
        files = {'audio': audio_file}
        response = requests.post(f"{API_BASE_URL}/api/predict", files=files, timeout=30)
        return response.json() if response.status_code == 200 else {'error': response.json().get('error', 'Prediction failed')}
    except Exception as e:
        return {'error': str(e)}

def upload_bulk_files(files):
    """Upload multiple files for retraining"""
    try:
        files_data = [('files', file) for file in files]
        response = requests.post(f"{API_BASE_URL}/api/upload_bulk", files=files_data, timeout=60)
        return response.json() if response.status_code == 200 else {'error': response.json().get('error', 'Upload failed')}
    except Exception as e:
        return {'error': str(e)}

def trigger_retrain():
    """Trigger model retraining"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/trigger_retrain", timeout=10)
        return response.json() if response.status_code == 200 else {'error': response.json().get('error', 'Retrain failed')}
    except Exception as e:
        return {'error': str(e)}

def get_retrain_status():
    """Get current retraining status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/retrain_status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_feature_analysis():
    """Get audio feature analysis"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/feature_analysis", timeout=30)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/model_performance", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

#  SIDEBAR NAVIGATION


with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Logo
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #a5b4fc; margin: 0;'>🐦 Bird Classifier</h2>
            <p style='color: #94a3b8; font-size: 0.9em; margin-top: 8px;'>ML-Powered Species ID</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Menu
    st.markdown("### 🧭 Navigation")
    
    if st.button("🎯 Predict Species", key="nav_predict", use_container_width=True):
        st.session_state.current_page = 'predict'
        st.rerun()
    
    if st.button("📦 Train Model", key="nav_train", use_container_width=True):
        st.session_state.current_page = 'train'
        st.rerun()
    
    if st.button("📊 View Analytics", key="nav_analytics", use_container_width=True):
        st.session_state.current_page = 'analytics'
        st.rerun()
    
    if st.button("⚙️ Performance", key="nav_performance", use_container_width=True):
        st.session_state.current_page = 'performance'
        st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 💻 System Status")
    health = get_system_health()
    
    if health:
        status_icon = "🟢" if health['status'] == 'healthy' else "🔴"
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin: 12px 0;'>
            <div style='font-size: 1.5em; text-align: center; margin-bottom: 8px;'>{status_icon}</div>
            <div style='text-align: center; color: #e0e0e0;'>
                <strong>Status:</strong> {health['status'].title()}<br>
                <strong>Version:</strong> {health['model_version']}<br>
                <strong>Uptime:</strong> {health['uptime_seconds']/3600:.1f}h
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ API Offline")
        st.caption(f"Target: {API_BASE_URL}")
    
    st.markdown("---")
    
    # Quick Stats
    metrics = get_metrics()
    if metrics:
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Predictions", metrics['total_predictions'])
        st.metric("Avg Confidence", f"{metrics['avg_confidence']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 16px 0; color: #94a3b8; font-size: 0.85em;'>
            <p><strong>ALU MLOps Project</strong></p>
            <p>Carine Ahishakiye</p>
            <p style='font-size: 0.8em; margin-top: 8px;'>2025</p>
        </div>
    """, unsafe_allow_html=True)

# MAIN HEADER

st.markdown("""
<div class='main-header'>
    <h1>🐦 Bird Sound Classification</h1>
    <p>Identify bird species from audio recordings using machine learning</p>
</div>
""", unsafe_allow_html=True)

# PAGE ROUTING


if st.session_state.current_page == 'predict':
    # PREDICT PAGE
    st.markdown("## 🎯 Species Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a bird sound recording (WAV or MP3)",
            type=['wav', 'mp3'],
            key='audio_upload'
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("Processing audio..."):
                    uploaded_file.seek(0)
                    result = make_prediction(uploaded_file)
                    
                    if 'error' in result:
                        st.markdown(f"<div class='error-box'>❌ Error: {result['error']}</div>", unsafe_allow_html=True)
                    else:
                        st.session_state.prediction_history.append(result)
                        
                        confidence = result['confidence']
                        conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                        
                        st.markdown(f"""
                        <div class='prediction-card'>
                            <h2 style='color: #1e40af; margin-bottom: 12px;'>
                                {conf_emoji} Predicted: <strong>{result['predicted_species']}</strong>
                            </h2>
                            <h3 style='color: #6366f1; margin-bottom: 8px;'>
                                Confidence: {result['confidence_percent']}
                            </h3>
                            <p style='color: #64748b; font-size: 0.9em;'>
                                Analyzed at {datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S on %b %d, %Y')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 📊 Top 5 Predictions")
                        top5_df = pd.DataFrame([
                            {"Species": species, "Confidence": f"{prob*100:.1f}%", "Score": prob}
                            for species, prob in result['top_5_predictions'].items()
                        ])
                        
                        fig = px.bar(
                            top5_df, x='Score', y='Species', orientation='h', text='Confidence',
                            color='Score', color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=320, showlegend=False, xaxis_title="Confidence",
                                        yaxis_title="", plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📖 How to Use")
        st.info("""
        **Step 1:** Upload an audio file containing bird sounds
        
        **Step 2:** Click "Analyze Audio" to start classification
        
        **Step 3:** Review the predicted species and confidence score
        
        **Tip:** Longer recordings (5-10 seconds) typically give better results
        """)
        
        if st.session_state.prediction_history:
            st.markdown("### 📜 Recent Results")
            for pred in st.session_state.prediction_history[-5:]:
                st.text(f"• {pred['predicted_species']}")
                st.caption(f"  {pred['confidence_percent']}")

elif st.session_state.current_page == 'train':
    # TRAIN PAGE
    st.markdown("## 📦 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Training Data")
        st.caption("Upload audio files to improve model accuracy")
        
        bulk_files = st.file_uploader(
            "Select audio files (WAV or MP3)",
            type=['wav', 'mp3'],
            accept_multiple_files=True,
            key='bulk_upload'
        )
        
        if bulk_files:
            st.success(f"✅ {len(bulk_files)} files ready")
            
            with st.expander("View files"):
                for i, file in enumerate(bulk_files[:10], 1):
                    st.text(f"{i}. {file.name} ({file.size/1024:.1f} KB)")
                if len(bulk_files) > 10:
                    st.caption(f"... and {len(bulk_files)-10} more")
            
            if st.button("📤 Upload to Server", use_container_width=True):
                with st.spinner("Uploading..."):
                    result = upload_bulk_files(bulk_files)
                    
                    if 'error' in result:
                        st.markdown(f"<div class='error-box'>❌ {result['error']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='success-box'>
                            ✅ Uploaded {result['uploaded_count']} files<br>
                            Total on server: {result['total_files_in_folder']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.uploaded_files_count = result['total_files_in_folder']
    
    with col2:
        st.markdown("### 🔄 Start Training")
        st.caption("Retrain the model with new data")
        
        retrain_status = get_retrain_status()
        
        if retrain_status and retrain_status['is_retraining']:
            st.markdown(f"<div class='info-box'>🔄 Training in progress: {retrain_status['progress']}%</div>", unsafe_allow_html=True)
            st.progress(retrain_status['progress'] / 100)
            
            if retrain_status['progress'] < 100:
                time.sleep(2)
                st.rerun()
        else:
            if st.session_state.uploaded_files_count >= 10:
                if st.button("🚀 Begin Training", type="primary", use_container_width=True):
                    with st.spinner("Starting..."):
                        result = trigger_retrain()
                        
                        if 'error' in result:
                            st.markdown(f"<div class='error-box'>❌ {result['error']}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='success-box'>✅ Training started with {result['files_count']} files</div>", unsafe_allow_html=True)
                            time.sleep(1)
                            st.rerun()
            else:
                st.warning(f"⚠️ Need at least 10 files (Current: {st.session_state.uploaded_files_count})")
        
        if retrain_status:
            st.markdown("### 📊 Training Info")
            if retrain_status['last_retrain']:
                last = datetime.fromisoformat(retrain_status['last_retrain'])
                st.info(f"Last trained: {last.strftime('%b %d at %H:%M')}")
            st.info(f"Current version: {retrain_status['model_version']}")

elif st.session_state.current_page == 'analytics':
    # ANALYTICS PAGE
    st.markdown("## 📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🎼 Show Audio Features", use_container_width=True):
            st.session_state.show_features = not st.session_state.get('show_features', False)
            st.rerun()
    
    metrics = get_metrics()
    
    if metrics and metrics['species_distribution']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🐦 Most Predicted Species")
            species_data = sorted(metrics['species_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
            species_df = pd.DataFrame(species_data, columns=['Species', 'Count'])
            
            fig = px.bar(species_df, x='Count', y='Species', orientation='h', color='Count',
                        color_continuous_scale='Blues', text='Count')
            fig.update_layout(height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Confidence Over Time")
            if metrics['recent_predictions']:
                conf_data = [{'Prediction': i+1, 'Confidence': pred['confidence']}
                           for i, pred in enumerate(metrics['recent_predictions'])]
                conf_df = pd.DataFrame(conf_data)
                
                fig = px.line(conf_df, x='Prediction', y='Confidence', markers=True, line_shape='spline')
                fig.update_layout(height=400, yaxis_range=[0, 1], yaxis_tickformat='.0%',
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions yet")
    
    if st.session_state.get('show_features', False):
        st.markdown("---")
        st.markdown("## 🎼 Audio Feature Analysis")
        
        with st.spinner("Analyzing features..."):
            features = get_feature_analysis()
            
            if features and 'error' not in features:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### MFCC")
                    if features['mfcc_analysis']['values']:
                        mfcc_df = pd.DataFrame({'Sample': features['mfcc_analysis']['labels'],
                                              'Value': features['mfcc_analysis']['values']})
                        fig = px.bar(mfcc_df, x='Sample', y='Value', color='Value', color_continuous_scale='Viridis')
                        fig.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(features['mfcc_analysis']['description'])
                
                with col2:
                    st.markdown("### Spectral Centroid")
                    if features['spectral_centroid']['values']:
                        sc_df = pd.DataFrame({'Sample': features['spectral_centroid']['labels'],
                                            'Value': features['spectral_centroid']['values']})
                        fig = px.bar(sc_df, x='Sample', y='Value', color='Value', color_continuous_scale='Reds')
                        fig.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(features['spectral_centroid']['description'])
                
                with col3:
                    st.markdown("### Zero Crossing Rate")
                    if features['zcr_analysis']['values']:
                        zcr_df = pd.DataFrame({'Sample': features['zcr_analysis']['labels'],
                                             'Value': features['zcr_analysis']['values']})
                        fig = px.bar(zcr_df, x='Sample', y='Value', color='Value', color_continuous_scale='Greens')
                        fig.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(features['zcr_analysis']['description'])
            else:
                st.warning("Upload training files to see audio features")

elif st.session_state.current_page == 'performance':
    # PERFORMANCE PAGE
    st.markdown("## ⚙️ Model Performance")
    
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()
    
    performance = get_model_performance()
    
    if performance and 'error' not in performance:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Predictions", performance['total_predictions'])
        with col2:
            st.metric("Avg Confidence", f"{performance['average_confidence']*100:.1f}%")
        with col3:
            st.metric("Median", f"{performance['median_confidence']*100:.1f}%")
        with col4:
            st.metric("Species Found", performance['unique_species_predicted'])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Confidence Breakdown")
            dist_data = pd.DataFrame([
                {'Level': 'High (>80%)', 'Count': performance['confidence_distribution']['high']},
                {'Level': 'Medium (50-80%)', 'Count': performance['confidence_distribution']['medium']},
                {'Level': 'Low (<50%)', 'Count': performance['confidence_distribution']['low']}
            ])
            
            fig = px.pie(dist_data, values='Count', names='Level',
                        color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top Species")
            if performance['top_predicted_species']:
                top_df = pd.DataFrame(performance['top_predicted_species'])
                fig = px.bar(top_df, x='count', y='species', orientation='h', color='count',
                           color_continuous_scale='Purples', text='count')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Confidence", f"{performance['min_confidence']*100:.1f}%")
        with col2:
            st.metric("Max Confidence", f"{performance['max_confidence']*100:.1f}%")
        with col3:
            st.metric("Std Dev", f"{performance['std_confidence']*100:.2f}%")
        
        st.info(f"Version {performance['model_version']} • Uptime: {performance['uptime_hours']:.1f} hours")
    else:
        st.warning("Make predictions to see performance data")