# Machine-Predictive-Maintenance-System
Industrial fault detection system using machine learning to predict equipment failures before they occur.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## Project Overview

This system uses machine learning to analyze sensor data and predict potential equipment faults in industrial machinery, enabling:
-  Reduced unexpected downtime
-  Lower maintenance costs
-  Optimized maintenance scheduling
-  Early fault detection

##  Features

- **Real-time Predictions**: Instant fault detection from sensor readings
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Binary Classification**: Normal vs Fault detection
- **Confidence Scores**: Prediction probability visualization
- **Historical Analytics**: Trend analysis and insights
- **18 Engineered Features**: Rolling statistics, lag features, time-based features

##  Technology Stack

- **Machine Learning**: Scikit-learn, Random Forest, SMOTE
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn
- **Web Interface**: Streamlit
- **Model Deployment**: Joblib

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-system.git
cd predictive-maintenance-system
```

2. **Create virtual environment** 
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib imbalanced-learn seaborn matplotlib
```

##  Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8000`

### Making Predictions

1. **Manual Input Mode**:
   - Use sliders to adjust sensor values
   - Click "Predict Fault Status"
   - View results and recommendations

### Example Input

```python
Vibration: 0.77 mm/s
Temperature: 107.5 °C
Pressure: 7.5 bar

Result: ⚠️ FAULT DETECTED
Recommendation: Schedule maintenance immediately
```

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 78.5% |
| Precision | 82.1% |
| Recall | 76.3% |
| F1-Score | 79.1% |

##  Project Structure

```
predictive-maintenance-system/
├── app.py                          # Streamlit dashboard
├── best_model_binary.pkl           # Trained Random Forest model
├── scaler_binary.pkl               # Feature scaler
├── feature_names_binary.pkl        # Feature list
├── preprocessing_config_binary.pkl # Preprocessing configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

##  Model Details

### Features Used (18 total)
- **Original Sensors**: Vibration, Temperature, Pressure
- **Rolling Statistics**: 5-point window means and standard deviations
- **Lag Features**: 1 and 2 time steps
- **Time Features**: Hour, Day of Week, Minute of Day

### Training Process
1. Time-series feature engineering
2. Train-test split (80-20)
3. Feature scaling (StandardScaler)
4. Class balancing (SMOTE)
5. Random Forest training
6. Cross-validation

##  Dataset

- **Source**: Industrial IoT Fault Detection Dataset (Kaggle)
- **Samples**: 1,000 sensor readings
- **Features**: 7 original features
- **Classes**: Binary (Normal vs Fault)

##  Academic Project

**Course**: Data Mining (Fall 2025)  
**Team Members**: 
- Ayesha Javed
- Syeda Maryam Fatima


##  Future Improvements

- [ ] Deep learning models (LSTM, GRU)
- [ ] Real-time streaming data support
- [ ] Multi-sensor fusion
- [ ] Anomaly detection algorithms
- [ ] Mobile app deployment
- [ ] API for integration

##  Known Issues

- Single prediction requires history generation for rolling features
- Limited to offline predictions (not connected to live sensors)

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


