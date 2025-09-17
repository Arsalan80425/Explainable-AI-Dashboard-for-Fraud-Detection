import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection XAI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        color: #c62828;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
        font-weight: bold;
    }
    .feature-value {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'current_transaction' not in st.session_state:
    st.session_state.current_transaction = None

# Load pre-trained model and artifacts
@st.cache_resource
def load_model_and_artifacts():
    """Load pre-trained model and related artifacts"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_metrics = joblib.load('model_metrics.pkl')
        
        return model, explainer, label_encoders, feature_names, model_metrics
    
    except FileNotFoundError as e:
        st.error(f"❌ Model artifacts not found: {str(e)}")
        st.error("""
        Please ensure the following files are in the same directory:
        - fraud_detection_model.pkl
        - shap_explainer.pkl  
        - label_encoders.pkl
        - feature_names.pkl
        - model_metrics.pkl
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.stop()

def prepare_features_for_prediction(instance_data, label_encoders, feature_names):
    """Prepare a single instance for prediction using saved encoders"""
    
    # Convert to DataFrame if it's a dict
    if isinstance(instance_data, dict):
        instance_df = pd.DataFrame([instance_data])
    else:
        instance_df = instance_data.copy()
    
    # Apply label encoding where needed
    for col in label_encoders.keys():
        if col in instance_df.columns:
            # Handle unseen categories by mapping them to a default value
            def safe_encode(x):
                try:
                    return label_encoders[col].transform([str(x)])[0]
                except ValueError:
                    # Unseen category - assign a default encoded value
                    return -1
            
            instance_df[col] = instance_df[col].apply(safe_encode)
    
    # Fill any missing columns with default values
    for col in feature_names:
        if col not in instance_df.columns:
            instance_df[col] = -999
    
    # Reorder columns to match training order
    instance_df = instance_df[feature_names]
    
    # Fill any remaining NaN values
    instance_df = instance_df.fillna(-999)
    
    return instance_df

def generate_counterfactual_explanation(instance, prediction_proba, model, feature_names):
    """Generate counterfactual explanations for predictions"""
    
    # Ensure instance is a numpy array
    if not isinstance(instance, np.ndarray):
        instance = np.array(instance)
    
    threshold = 0.11
    is_fraud = prediction_proba > threshold
    
    suggestions = []
    
    if is_fraud:
        st.markdown("### 🔄 How to make this transaction legitimate:")
        
        # Transaction amount suggestions
        if 'TransactionAmt' in feature_names:
            amt_idx = feature_names.index('TransactionAmt')
            current_amt = instance[amt_idx]
            
            # Try different amounts
            test_amounts = [current_amt * 0.5, current_amt * 0.7, 50, 100, 200]
            for test_amt in test_amounts:
                if test_amt != current_amt and test_amt > 0:
                    modified = instance.copy()
                    modified[amt_idx] = test_amt
                    # Reshape for prediction
                    modified_2d = modified.reshape(1, -1)
                    new_prob = model.predict_proba(modified_2d)[0, 1]
                    if new_prob < threshold:
                        suggestions.append(f"💰 Reducing transaction amount to ${test_amt:.2f} would likely result in approval (from ${current_amt:.2f})")
                        break
        
        # Transaction hour suggestions
        if 'TransactionHour' in feature_names:
            hour_idx = feature_names.index('TransactionHour')
            current_hour = instance[hour_idx]
            
            # Try business hours
            business_hours = [9, 10, 11, 14, 15, 16]
            for hour in business_hours:
                if hour != current_hour:
                    modified = instance.copy()
                    modified[hour_idx] = hour
                    modified_2d = modified.reshape(1, -1)
                    new_prob = model.predict_proba(modified_2d)[0, 1]
                    if new_prob < threshold:
                        suggestions.append(f"🕐 Processing during business hours ({hour}:00) instead of {int(current_hour)}:00 would reduce fraud risk")
                        break
        
        # Day of week suggestions
        if 'TransactionDayOfWeek' in feature_names:
            dow_idx = feature_names.index('TransactionDayOfWeek')
            current_dow = instance[dow_idx]
            
            # Try weekdays (Monday=0 to Friday=4)
            weekdays = [0, 1, 2, 3, 4]
            for dow in weekdays:
                if dow != current_dow:
                    modified = instance.copy()
                    modified[dow_idx] = dow
                    modified_2d = modified.reshape(1, -1)
                    new_prob = model.predict_proba(modified_2d)[0, 1]
                    if new_prob < threshold:
                        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        suggestions.append(f"📅 Processing on {day_names[int(dow)]} instead of {day_names[int(current_dow)]} would reduce risk")
                        break
        
        # Email domain suggestions
        if 'P_emaildomain' in feature_names:
            email_idx = feature_names.index('P_emaildomain')
            current_email = instance[email_idx]
            
            # Try common legitimate email domains (encoded values may vary)
            if current_email == -1 or current_email > 10:  # Assuming higher values are suspicious
                modified = instance.copy()
                modified[email_idx] = 1  # Assuming this represents a common domain like gmail
                modified_2d = modified.reshape(1, -1)
                new_prob = model.predict_proba(modified_2d)[0, 1]
                if new_prob < threshold:
                    suggestions.append("📧 Using a common email domain (like Gmail, Yahoo) would significantly reduce fraud risk")
        
        if not suggestions:
            suggestions.append("⚠️ Multiple high-risk factors detected. Consider additional verification methods or manual review.")
    
    else:
        suggestions.append("✅ Transaction patterns match legitimate behavior")
        suggestions.append("✅ All risk factors are within acceptable thresholds")
        
        # Show what could make it risky
        if 'TransactionAmt' in feature_names:
            amt_idx = feature_names.index('TransactionAmt')
            current_amt = instance[amt_idx]
            
            # Test higher amounts
            high_amount = current_amt * 3
            modified = instance.copy()
            modified[amt_idx] = high_amount
            modified_2d = modified.reshape(1, -1)
            new_prob = model.predict_proba(modified_2d)[0, 1]
            if new_prob > threshold:
                suggestions.append(f"ℹ️ Increasing amount to ${high_amount:.2f} would trigger fraud alerts")
    
    return suggestions

def explain_prediction_with_shap(instance, model, explainer, feature_names, instance_idx=0):
    """Generate detailed SHAP explanations for a prediction"""
    
    # Ensure instance is in the correct format (2D array)
    if isinstance(instance, (list, np.ndarray)) and len(instance.shape) == 1:
        instance_2d = instance.reshape(1, -1)
    else:
        instance_2d = np.array([instance]) if not isinstance(instance, np.ndarray) else instance
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(instance_2d)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # For binary classification, take positive class (fraud)
        shap_values = shap_values[1]
    
    # Ensure we have the right shape
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]  # Take first instance if batch
    
    # Create explanation DataFrame
    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': instance.flatten() if hasattr(instance, 'flatten') else instance,
        'SHAP_Value': shap_values,
        'Abs_SHAP': np.abs(shap_values)
    }).sort_values('Abs_SHAP', ascending=False)
    
    return explanation_df, shap_values

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🛡️ Explainable AI Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("### Advanced Fraud Detection with Full Explainability")
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Navigation")
        page = st.radio("Select Page", 
                       ["🏠 Model Overview", "🔍 Transaction Analysis", 
                        "📊 Model Performance", "ℹ️ About"])
        
        st.markdown("---")
        
        # Model status
        st.header("📈 Model Status")
        if st.session_state.model is not None:
            st.success("✅ Pre-trained Model Ready")
            st.metric("Features", len(st.session_state.feature_names))
            if st.session_state.model_metrics:
                st.metric("ROC-AUC", f"{st.session_state.model_metrics['roc_auc']:.4f}")
        else:
            st.info("⏳ Loading Model...")
    
    # Load model if not already done
    if st.session_state.model is None:
        with st.spinner("🔄 Loading pre-trained model and artifacts..."):
            # Load pre-trained model and artifacts
            try:
                model, explainer, label_encoders, feature_names, model_metrics = load_model_and_artifacts()
                
                # Store results in session state
                st.session_state.model = model
                st.session_state.explainer = explainer
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_names = feature_names
                st.session_state.model_metrics = model_metrics
                
                st.success("✅ Pre-trained model loaded successfully!")
                
                # Display model metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROC-AUC", f"{model_metrics['roc_auc']:.4f}")
                with col2:
                    st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                with col3:
                    st.metric("F1-Score", f"{model_metrics['f1_score']:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()
    
    # Page routing
    if page == "🏠 Model Overview":
        show_model_overview()
    elif page == "🔍 Transaction Analysis":
        show_transaction_analysis()
    elif page == "📊 Model Performance":
        show_model_performance()
    else:
        show_about()

def show_model_overview():
    """Display model overview and global explanations"""
    st.header("🌍 Pre-trained Model Overview")
    
    if st.session_state.model_metrics:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Model Type", "LightGBM")
        with col2:
            st.metric("ROC-AUC", f"{st.session_state.model_metrics['roc_auc']:.4f}")
        with col3:
            st.metric("Precision", f"{st.session_state.model_metrics['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{st.session_state.model_metrics['recall']:.4f}")
        with col5:
            st.metric("F1-Score", f"{st.session_state.model_metrics['f1_score']:.4f}")
    
    st.markdown("---")
    
    # Feature importance analysis
    if st.session_state.model is not None:
        st.subheader("🎯 Feature Importance Analysis")
        
        try:
            # Get feature importance from the model
            if hasattr(st.session_state.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': st.session_state.feature_names,
                    'importance': st.session_state.model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig = px.bar(importance_df, 
                            x='importance', 
                            y='feature', 
                            orientation='h',
                            title="Top 20 Most Important Features",
                            color='importance',
                            color_continuous_scale='Blues')
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Could not display feature importance. This might be expected depending on your model type.")
    
    # Model insights
    with st.expander("🧠 Model Insights & Interpretation"):
        st.markdown("""
        ### Pre-trained Model Information:
        
        **📁 Model Artifacts Loaded:**
        - **Pre-trained LightGBM Model**: Ready for fraud detection
        - **SHAP Explainer**: For generating feature explanations
        - **Label Encoders**: For processing categorical features
        - **Feature Names**: Maintaining consistency with training data
        - **Model Metrics**: Performance benchmarks
        
        **🎯 Key Capabilities:**
        - **Real-time Predictions**: Instant fraud scoring for transactions
        - **Explainable Results**: SHAP-based feature contribution analysis
        - **Counterfactual Analysis**: What-if scenario modeling
        
        **⚡ Performance Highlights:**
        - High precision reduces false positives (legitimate transactions blocked)
        - Good recall ensures most fraudulent transactions are caught
        - Optimized for production deployment
        - Consistent feature engineering pipeline
        """)

def show_transaction_analysis():
    """Interactive transaction analysis with full explanations"""
    st.header("🔍 Individual Transaction Analysis")
    
    st.markdown("""
    Analyze individual transactions with comprehensive explanations including:
    - **Prediction confidence** with visual risk gauge
    - **SHAP explanations** showing feature contributions
    - **Counterfactual analysis** suggesting how to change the outcome
    """)
    
    # Analysis options
    analysis_type = st.radio(
        "Choose analysis method:", 
        ["🎯 Manual Input", "🎲 Random Entry"],
        horizontal=True
    )
    
    # Transaction selection - Manual Input or Random Entry
    st.subheader("📝 Enter Transaction Feature Values")
    
    if analysis_type == "🎲 Random Entry":
        # Generate random transaction data
        if st.button("🎲 Generate Random Transaction"):
            # Create random transaction data
            random_data = {
                'TransactionAmt': np.random.randint(10, 1000),
                'TransactionHour': np.random.randint(0, 24),
                'TransactionDayOfWeek': np.random.randint(0, 7),
                'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R']),
                'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover']),
                'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'other'])
            }
            
            # Store in session state
            st.session_state.current_transaction = {
                'data': random_data,
                'label': 'Random Entry',
                'index': 'Random'
            }
            st.success("✅ Random transaction generated! Ready for analysis.")
    
    # Manual input form (only show for manual input option)
    if analysis_type == "🎯 Manual Input":
        with st.form("transaction_input"):
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_amt = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.0,
                    value=150.0,
                    step=1.0,
                    help="Enter the transaction amount in USD"
                )
                
                transaction_hour = st.slider(
                    "Transaction Hour (0-23)",
                    min_value=0,
                    max_value=23,
                    value=14,
                    help="Hour of the day when transaction occurred"
                )
                
                transaction_dow = st.selectbox(
                    "Day of Week",
                    options=[0, 1, 2, 3, 4, 5, 6],
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                    index=2,
                    help="Day of the week (0=Monday, 6=Sunday)"
                )
            
            with col2:
                product_cd = st.selectbox(
                    "Product Code",
                    options=['W', 'H', 'C', 'S', 'R'],
                    index=0,
                    help="Product category code"
                )
                
                card4 = st.selectbox(
                    "Card Type",
                    options=['visa', 'mastercard', 'american express', 'discover'],
                    index=0,
                    help="Credit card network"
                )
                
                p_emaildomain = st.selectbox(
                    "Email Domain",
                    options=['gmail.com', 'yahoo.com', 'hotmail.com', 'other'],
                    index=0,
                    help="Email domain of purchaser"
                )
            
            submitted = st.form_submit_button("🔍 Analyze Transaction")
            
            if submitted:
                # Create transaction data dictionary
                input_data = {
                    'TransactionAmt': transaction_amt,
                    'TransactionHour': transaction_hour,
                    'TransactionDayOfWeek': transaction_dow,
                    'ProductCD': product_cd,
                    'card4': card4,
                    'P_emaildomain': p_emaildomain
                }
                
                try:
                    # Store in session state
                    st.session_state.current_transaction = {
                        'data': input_data,
                        'label': 'Manual Input',
                        'index': 'Manual'
                    }
                    st.success("✅ Manual transaction ready for analysis!")
                
                except Exception as e:
                    st.error(f"❌ Error preparing manual input: {str(e)}")
    
    # Display current transaction details if available
    if st.session_state.current_transaction is not None:
        transaction = st.session_state.current_transaction
        st.info(f"**Current Transaction:** {transaction['label']}")
        
        # Show transaction details in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Transaction Details:**")
            for key, value in transaction['data'].items():
                st.write(f"**{key}:** {value}")
    
    # Analyze selected transaction
    if st.session_state.current_transaction is not None:
        transaction = st.session_state.current_transaction
        raw_data = transaction['data']
        actual_label = transaction['label']
        idx = transaction['index']
        
        st.markdown("---")
        st.subheader(f"📋 Transaction Analysis (Source: {idx})")
        
        try:
            # Prepare data for prediction
            prepared_data = prepare_features_for_prediction(
                raw_data, 
                st.session_state.label_encoders, 
                st.session_state.feature_names
            )
            
            instance = prepared_data.iloc[0].values
            
            # Make prediction
            prediction_proba = st.session_state.model.predict_proba([instance])[0]
            prediction = int(prediction_proba[1] > 0.5)
            
            # Display prediction results
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # Prediction result
                if prediction == 1:
                    st.markdown('<div class="fraud-alert">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
                    st.error(f"🚨 Fraud Probability: **{prediction_proba[1]:.1%}**")
                else:
                    st.markdown('<div class="safe-alert">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)
                    st.success(f"✅ Fraud Probability: **{prediction_proba[1]:.1%}**")
            
            with col2:
                # Input data summary
                st.info("**Transaction Details:**")
                for key, value in raw_data.items():
                    st.text(f"{key}: {value}")
            
            with col3:
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba[1] * 100,
                    title={'text': "Risk Score %"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Analysis
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation: Why This Prediction?")
            
            # Calculate SHAP values for this instance
            explanation_df, shap_values = explain_prediction_with_shap(
                instance, st.session_state.model, st.session_state.explainer, 
                st.session_state.feature_names
            )
            
            # Top contributing features
            top_features = explanation_df.head(15)
            
            # Waterfall chart showing feature contributions
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["relative"] * len(top_features),
                x=[f.replace('_', ' ')[:20] for f in top_features['Feature']],
                y=top_features['SHAP_Value'],
                text=[f"{v:.4f}" for v in top_features['SHAP_Value']],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#ff6b6b"}},
                decreasing={"marker": {"color": "#4ecdc4"}},
                totals={"marker": {"color": "blue"}}
            ))
            
            fig.update_layout(
                title="Feature Contributions to Fraud Score (SHAP Values)",
                height=500,
                showlegend=False,
                xaxis_tickangle=-45,
                xaxis_title="Features",
                yaxis_title="SHAP Value (Impact on Prediction)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature values and explanations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Key Feature Values")
                for _, row in top_features.head(8).iterrows():
                    impact_color = "🔴" if row['SHAP_Value'] > 0 else "🟢"
                    impact_text = "increases" if row['SHAP_Value'] > 0 else "decreases"
                    
                    st.markdown(f"""
                    <div class="feature-value" style="background-color: white; color: black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>{row['Feature']}</strong><br>
                    Value: {row['Value']:.3f} | Impact: {impact_color} {row['SHAP_Value']:.4f}<br>
                    <em>This feature {impact_text} fraud probability</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📈 SHAP Value Distribution")
                
                # Bar chart of SHAP values
                colors = ['red' if x > 0 else 'green' for x in top_features.head(8)['SHAP_Value']]
                
                fig = go.Figure(go.Bar(
                    x=top_features.head(8)['SHAP_Value'],
                    y=[f.replace('_', ' ')[:25] for f in top_features.head(8)['Feature']],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in top_features.head(8)['SHAP_Value']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Top 8 Feature Impacts",
                    height=400,
                    xaxis_title="SHAP Value",
                    yaxis_title="Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Counterfactual Analysis
            st.markdown("---")
            st.subheader("🔄 Counterfactual Analysis")
            
            st.markdown("""
            **What if scenarios:** Understanding how small changes could alter the prediction.
            """)
            
            counterfactuals = generate_counterfactual_explanation(
                instance, prediction_proba[1], st.session_state.model, st.session_state.feature_names
            )
            
            for i, cf in enumerate(counterfactuals, 1):
                if "✅" in cf:
                    st.success(cf)
                elif "⚠️" in cf or "❌" in cf:
                    st.error(cf)
                elif "ℹ️" in cf:
                    st.info(cf)
                else:
                    st.warning(cf)
            
            # Detailed feature analysis
            with st.expander("🔍 Detailed Feature Analysis"):
                st.subheader("All Feature Values and Impacts")
                
                # Create a comprehensive table
                display_df = explanation_df.copy()
                display_df['Impact_Direction'] = display_df['SHAP_Value'].apply(
                    lambda x: "Increases Fraud Risk" if x > 0 else "Decreases Fraud Risk"
                )
                display_df['Impact_Magnitude'] = display_df['SHAP_Value'].apply(
                    lambda x: "High" if abs(x) > 0.1 else "Medium" if abs(x) > 0.01 else "Low"
                )
                
                # Format for display
                display_df['Feature'] = display_df['Feature'].str.replace('_', ' ').str.title()
                display_df['Value'] = display_df['Value'].round(4)
                display_df['SHAP_Value'] = display_df['SHAP_Value'].round(6)
                
                st.dataframe(
                    display_df[['Feature', 'Value', 'SHAP_Value', 'Impact_Direction', 'Impact_Magnitude']],
                    use_container_width=True,
                    hide_index=True
                )
            
        except Exception as e:
            st.error(f"❌ Error analyzing transaction: {str(e)}")
            st.error("This might be due to missing features or encoding issues.")

def show_model_performance():
    """Display model performance with available metrics"""
    st.header("📊 Model Performance Analysis")
    
    if st.session_state.model_metrics:
        # Performance metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ROC-AUC Score", f"{st.session_state.model_metrics['roc_auc']:.4f}")
        with col2:
            st.metric("Accuracy", f"{st.session_state.model_metrics['accuracy']:.4f}")
        with col3:
            st.metric("F1-Score", f"{st.session_state.model_metrics['f1_score']:.4f}")
        with col4:
            st.metric("Precision", f"{st.session_state.model_metrics['precision']:.4f}")
        
        st.markdown("---")
        
        # Performance visualization
        st.subheader("📈 Performance Metrics Comparison")
        
        metrics_data = {
            'Metric': ['ROC-AUC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'],
            'Value': [
                st.session_state.model_metrics['roc_auc'],
                st.session_state.model_metrics['accuracy'],
                st.session_state.model_metrics['f1_score'],
                st.session_state.model_metrics['precision'],
                st.session_state.model_metrics['recall']
            ]
        }
        
        fig = px.bar(
            x=metrics_data['Value'],
            y=metrics_data['Metric'],
            orientation='h',
            title="Model Performance Metrics",
            color=metrics_data['Value'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance interpretation
        st.subheader("🎯 Performance Interpretation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Strengths")
            if st.session_state.model_metrics['roc_auc'] > 0.9:
                st.success("🎉 Excellent discrimination capability (ROC-AUC > 0.9)")
            elif st.session_state.model_metrics['roc_auc'] > 0.8:
                st.info("👍 Good discrimination capability (ROC-AUC > 0.8)")
            
            if st.session_state.model_metrics['precision'] > 0.8:
                st.success("✅ High precision - low false positive rate")
            
            if st.session_state.model_metrics['recall'] > 0.8:
                st.success("✅ High recall - catches most fraud cases")
        
        with col2:
            st.markdown("### Business Impact")
            
            # Estimated business metrics
            if st.session_state.model_metrics['precision'] > 0.85:
                st.metric("Customer Friction", "Low", delta="Good")
                st.caption("Few legitimate transactions blocked")
            
            if st.session_state.model_metrics['recall'] > 0.8:
                st.metric("Fraud Detection", "High", delta="Excellent")
                st.caption("Most fraudulent transactions caught")
            
            overall_score = (st.session_state.model_metrics['precision'] + st.session_state.model_metrics['recall']) / 2
            if overall_score > 0.8:
                st.metric("Overall Performance", "Excellent", delta="Production Ready")
    else:
        st.warning("Model metrics not available.")

def show_about():
    """Show information about the dashboard"""
    st.header("ℹ️ About This Pre-trained Model Dashboard")
    
    st.markdown("""
    ## 🛡️ Explainable AI Fraud Detection System
    
    This dashboard demonstrates advanced fraud detection with full explainability capabilities using a **pre-trained model**.
    Built for financial institutions that need both high-performance fraud detection and 
    complete transparency in AI decision-making.
    
    ### 🎯 Key Features
    
    #### 🤖 Pre-trained Machine Learning
    - **LightGBM Model**: Pre-trained gradient boosting model optimized for fraud detection
    - **Ready-to-Deploy**: No training required - instant predictions
    - **Consistent Performance**: Validated model with known performance metrics
    - **Production-Ready**: Optimized for real-world deployment
    
    #### 🔍 Explainability
    - **SHAP Values**: Global and local feature importance explanations
    - **Counterfactual Explanations**: "What-if" scenario analysis
    - **Visual Explanations**: Intuitive charts and gauges
    - **Business Context**: Understand the reasoning behind predictions
    
    #### 📊 Analysis Capabilities
    - **Individual Transaction Analysis**: Deep-dive explanations for single transactions
    - **Manual Input**: Test custom transaction scenarios
    - **Performance Monitoring**: Track model effectiveness
    
    ### 🏗️ Technical Architecture
    
    #### Pre-trained Model Components
    1. **fraud_detection_model.pkl**: Trained LightGBM classifier
    2. **shap_explainer.pkl**: SHAP TreeExplainer for feature importance
    3. **label_encoders.pkl**: Preprocessing encoders for categorical features
    4. **feature_names.pkl**: Feature names and order consistency
    5. **model_metrics.pkl**: Performance benchmarks and evaluation metrics
    
    #### Data Processing Pipeline
    1. **Feature Preparation**: Apply consistent encoding and preprocessing
    2. **Prediction**: Use pre-trained model for instant fraud scoring
    3. **Explanation**: Generate SHAP-based explanations
    
    ### 💼 Business Value
    
    #### For Financial Institutions
    - **Instant Deployment**: No training time required
    - **Regulatory Compliance**: Explainable AI for audit requirements
    - **Customer Trust**: Transparent fraud decisions
    - **Operational Efficiency**: Reduced false positives
    - **Risk Management**: Proven fraud detection accuracy
    
    #### For Data Scientists
    - **Model Interpretability**: Understand feature contributions
    - **Rapid Prototyping**: Test scenarios without retraining
    - **Performance Analysis**: Evaluate model effectiveness
    - **Counterfactual Analysis**: Advanced explanation techniques
    
    ### 🚀 Getting Started
    
    #### Required Files
    Make sure you have these files in your working directory:
    - `fraud_detection_model.pkl` - Pre-trained model
    - `shap_explainer.pkl` - SHAP explainer
    - `label_encoders.pkl` - Feature encoders
    - `feature_names.pkl` - Feature names list
    - `model_metrics.pkl` - Model performance metrics
    
    #### Quick Start Steps
    1. **Load Model**: Pre-trained artifacts load automatically
    2. **Analyze Transactions**: Test individual predictions
    3. **Explore Results**: Use SHAP explanations and counterfactuals
    
    ---
    
    **Built for transparent and responsible AI in financial services by Arsalan**
    """)

if __name__ == "__main__":
    main()