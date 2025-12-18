import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==================================================================== 
# PAGE CONFIG
# ==================================================================== 
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ==================================================================== 
# CUSTOM CSS
# ==================================================================== 
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-healthy {
        background: #2ecc71;
        color: white;
    }
    .status-critical {
        background: #e74c3c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================== 
# LOAD MODELS
# ==================================================================== 
@st.cache_resource
def load_models():
    """Load all saved models and configurations"""
    try:
        model = joblib.load('best_model_binary.pkl')
        scaler = joblib.load('scaler_binary.pkl')
        feature_names = joblib.load('feature_names_binary.pkl')
        preprocessing_config = joblib.load('preprocessing_config_binary.pkl')
        return model, scaler, feature_names, preprocessing_config
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Make sure these files exist: best_model_binary.pkl, scaler_binary.pkl, feature_names_binary.pkl, preprocessing_config_binary.pkl")
        return None, None, None, None

model, scaler, feature_names, preprocessing_config = load_models()

# ==================================================================== 
# FEATURE ENGINEERING
# ==================================================================== 
def engineer_features(df_input):
    """Apply the same feature engineering as training"""
    df = df_input.copy()
    
    # Ensure timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Time features
    if 'Timestamp' in df.columns:
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['MinuteOfDay'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
    
    # Sensor features
    sensors = preprocessing_config['numerical_sensors']
    
    for sensor in sensors:
        if sensor in df.columns:
            # Rolling statistics
            df[f'{sensor}_rolling_mean_5'] = df[sensor].rolling(5, min_periods=1).mean()
            df[f'{sensor}_rolling_std_5'] = df[sensor].rolling(5, min_periods=1).std()
            
            # Lag features
            df[f'{sensor}_lag_1'] = df[sensor].shift(1)
            df[f'{sensor}_lag_2'] = df[sensor].shift(2)
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def prepare_features(df_input):
    """Prepare features exactly as in training"""
    df = engineer_features(df_input)
    
    features_to_drop = preprocessing_config['features_to_drop']
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    
    df_features = df.drop(columns=features_to_drop, errors='ignore')
    
    # Ensure all required features exist
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    
    # Select only training features in correct order
    df_features = df_features[feature_names]
    return df_features

# ==================================================================== 
# HEADER
# ==================================================================== 
st.markdown('<h1 class="main-title">🔧 Predictive Maintenance System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Industrial Fault Detection</p>', unsafe_allow_html=True)

if model is None:
    st.stop()

# ==================================================================== 
# SIDEBAR
# ==================================================================== 
st.sidebar.markdown("## 📊 Control Panel")
st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "Select Input Mode:",
    ["🎚️ Manual Input"],
    help="Choose how to input sensor data"
)

prediction_result = None
confidence_scores = None

if input_mode == "🎚️ Manual Input":
    st.sidebar.markdown("### Sensor Readings")
    
    vibration = st.sidebar.slider(
        "Vibration (mm/s)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Machine vibration intensity"
    )
    
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=50.0,
        max_value=130.0,
        value=90.0,
        step=0.5,
        help="Operating temperature"
    )
    
    pressure = st.sidebar.slider(
        "Pressure (bar)",
        min_value=7.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
        help="System pressure"
    )
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔮 Predict Fault Status", type="primary", use_container_width=True):
     with st.spinner("🔄 Analyzing..."):
        try:
            # --------------------------------------------------
            # CREATE REALISTIC SHORT TIME-SERIES HISTORY
            # --------------------------------------------------
            np.random.seed(None)

            timestamps = pd.date_range(
                end=datetime.now(),
                periods=5,
                freq="1min"
            )

            # Add small realistic noise (sensor drift)
            vibration_series = vibration + np.random.normal(0, 0.03, size=5)
            temperature_series = temperature + np.random.normal(0, 1.5, size=5)
            pressure_series = pressure + np.random.normal(0, 0.15, size=5)

            history_data = pd.DataFrame({
                'Timestamp': timestamps,
                'Vibration (mm/s)': vibration_series,
                'Temperature (°C)': temperature_series,
                'Pressure (bar)': pressure_series
            })

            # --------------------------------------------------
            # FEATURE PREPARATION (SAME AS TRAINING)
            # --------------------------------------------------
            X = prepare_features(history_data)

            # Use ONLY latest time step
            X_current = X.iloc[[-1]]

            # Scale using training scaler
            X_scaled = scaler.transform(X_current)

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------
            prediction_result = model.predict(X_scaled)[0]

            if hasattr(model, 'predict_proba'):
                confidence_scores = model.predict_proba(X_scaled)[0]

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)


# ==================================================================== 
# MAIN TABS
# ==================================================================== 
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Analytics", "ℹ️ About"])

# ==================================================================== 
# TAB 1: PREDICTION
# ==================================================================== 
with tab1:
    if prediction_result is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Prediction Result")
            
            if prediction_result == 0:
                st.markdown(
                    '<div class="status-badge status-healthy">✅ NORMAL OPERATION</div>',
                    unsafe_allow_html=True
                )
                st.success("System is operating within normal parameters")
                risk_level = "Low"
                risk_color = "#2ecc71"
                recommendation = """
                ✓ Continue normal operations
                ✓ Next inspection: As scheduled
                ✓ Maintenance: Not required
                """
            else:
                st.markdown(
                    '<div class="status-badge status-critical">⚠️ FAULT DETECTED</div>',
                    unsafe_allow_html=True
                )
                st.error("Fault detected - Maintenance required!")
                risk_level = "High"
                risk_color = "#e74c3c"
                recommendation = """
                🚨 Schedule maintenance immediately
                🚨 Inspect all components
                🚨 Monitor system closely
                🚨 Document findings
                """
            
            st.markdown(f"**Risk Level:** {risk_level}")
            
        with col2:
            if confidence_scores is not None:
                st.markdown("### 📊 Confidence Scores")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Normal', 'Fault'],
                        y=confidence_scores * 100,
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[f'{score*100:.1f}%' for score in confidence_scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Confidence (%)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 📋 Recommendations")
        st.info(recommendation)
        
        # Sensor readings
        st.markdown("### 📊 Current Sensor Readings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🌊 Vibration",
                value=f"{vibration:.2f} mm/s",
                delta=f"{(vibration - 0.5):.2f}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="🌡️ Temperature",
                value=f"{temperature:.1f} °C",
                delta=f"{(temperature - 90):.1f}",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="💨 Pressure",
                value=f"{pressure:.1f} bar",
                delta=f"{(pressure - 8.5):.1f}",
                delta_color="inverse"
            )
        
        # Gauges
        st.markdown("### 🎚️ Real-Time Gauges")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_vib = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vibration,
                title={'text': "Vibration (mm/s)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig_vib.update_layout(height=250)
            st.plotly_chart(fig_vib, use_container_width=True)
        
        with col2:
            fig_temp = go.Figure(go.Indicator(
                mode="gauge+number",
                value=temperature,
                title={'text': "Temperature (°C)"},
                gauge={
                    'axis': {'range': [50, 130]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [50, 80], 'color': "lightgreen"},
                        {'range': [80, 110], 'color': "yellow"},
                        {'range': [110, 130], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 120
                    }
                }
            ))
            fig_temp.update_layout(height=250)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col3:
            fig_press = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pressure,
                title={'text': "Pressure (bar)"},
                gauge={
                    'axis': {'range': [7, 10]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [7, 8], 'color': "lightgreen"},
                        {'range': [8, 9], 'color': "yellow"},
                        {'range': [9, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9.5
                    }
                }
            ))
            fig_press.update_layout(height=250)
            st.plotly_chart(fig_press, use_container_width=True)
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2>👋 Welcome to the Predictive Maintenance System</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Use the sidebar to input sensor readings and get real-time fault predictions
            </p>
            <p style='margin-top: 2rem;'>
                ⚙️ Adjust sensor values<br>
                🔮 Click "Predict Fault Status"<br>
                📊 View detailed results
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================================== 
# TAB 2: ANALYTICS
# ==================================================================== 
with tab2:
    st.markdown("## 📈 System Analytics")
    
    # Sample historical data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'Vibration': np.random.normal(0.5, 0.15, len(dates)),
        'Temperature': np.random.normal(90, 15, len(dates)),
        'Pressure': np.random.normal(8.5, 0.5, len(dates)),
        'Faults': np.random.choice([0, 1], len(dates), p=[0.7, 0.3])
    })
    
    # Time series
    st.markdown("### 📉 Sensor Trends (Last Year)")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Vibration', 'Temperature', 'Pressure'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data['Date'], y=historical_data['Vibration'],
                  mode='lines', name='Vibration', line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data['Date'], y=historical_data['Temperature'],
                  mode='lines', name='Temperature', line=dict(color='#e74c3c', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data['Date'], y=historical_data['Pressure'],
                  mode='lines', name='Pressure', line=dict(color='#2ecc71', width=2)),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="mm/s", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="bar", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Fault distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Fault Distribution")
        fault_counts = historical_data['Faults'].value_counts().sort_index()
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=['Normal', 'Fault'],
                values=fault_counts.values,
                marker_colors=['#2ecc71', '#e74c3c'],
                hole=0.4
            )
        ])
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Monthly Faults")
        monthly_faults = historical_data.groupby(
            historical_data['Date'].dt.month
        )['Faults'].sum()
        
        fig_bar = go.Figure(data=[
            go.Bar(x=monthly_faults.index, y=monthly_faults.values,
                  marker_color='#667eea')
        ])
        fig_bar.update_layout(
            xaxis_title="Month",
            yaxis_title="Fault Count",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==================================================================== 
# TAB 3: ABOUT
# ==================================================================== 
with tab3:
    st.markdown("## ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        Binary classification system for industrial fault detection:
        
        - **Class 0**: Normal Operation ✅
        - **Class 1**: Fault Detected ⚠️
        
        ### 🛠️ Technology Stack
        
        - Machine Learning: Random Forest
        - Data Processing: Pandas, NumPy
        - Visualization: Plotly
        - Interface: Streamlit
        - Balancing: SMOTE
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Performance
        
        Expected metrics with binary classification:
        - Accuracy: 70-85%
        - Precision: 75-90%
        - Recall: 70-85%
        - F1-Score: 72-87%
        
        ### 👥 Team
        
        **Course**: Data Mining (Fall 2025)
        **Members**: Ayesha Javed, Syeda Maryam Fatima
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Features Used")
    
    st.markdown("""
    **Original Sensors:**
    - Vibration (mm/s)
    - Temperature (°C)
    - Pressure (bar)
    
    **Engineered Features:**
    - Rolling mean (5-point window)
    - Rolling standard deviation
    - Lag features (1 and 2 time steps)
    - Time-based features (Hour, Day, Minute)
    
    **Total Features**: 18
    """)

# ==================================================================== 
# FOOTER
# ==================================================================== 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Predictive Maintenance System v2.0 (Binary Classification)</strong></p>
    <p>Data Mining Project | Fall 2025</p>
</div>
""", unsafe_allow_html=True)