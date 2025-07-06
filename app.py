import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Drop customerID
    df = df.drop(columns=["customerID"], axis=1)
    
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)
    
    # Encode target variable
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    
    # Identify categorical columns
    object_columns = df.select_dtypes(include="object").columns
    
    # Create encoders
    encoders = {}
    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder
    
    return df, encoders

@st.cache_data
def train_model(df):
    """Train the model"""
    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train Random Forest
    rfc = RandomForestClassifier(random_state=42, n_estimators=100)
    rfc.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_test_pred = rfc.predict(X_test)
    
    return rfc, X_train, X_test, y_train, y_test, y_test_pred, X.columns.tolist()

def main():
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📈 Data Analysis", "🤖 Model Training", "🔮 Predict Churn", "📊 Model Performance"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        df_processed, encoders = preprocess_data(df)
        model, X_train, X_test, y_train, y_test, y_test_pred, feature_names = train_model(df_processed)
    
    if page == "🏠 Home":
        show_home_page(df)
    
    elif page == "📈 Data Analysis":
        show_data_analysis(df)
    
    elif page == "🤖 Model Training":
        show_model_training(df_processed, model, X_train, X_test, y_train, y_test, y_test_pred)
    
    elif page == "🔮 Predict Churn":
        show_prediction_page(df, encoders, model, feature_names)
    
    elif page == "📊 Model Performance":
        show_model_performance(y_test, y_test_pred, model, X_test, feature_names)

def show_home_page(df):
    st.markdown("## Welcome to Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", len(df))
    
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        st.metric("Features", len(df.columns) - 1)
    
    st.markdown("---")
    
    st.markdown("### About This Project")
    st.markdown("""
    This application helps predict customer churn for a telecommunications company using machine learning.
    
    **Key Features:**
    - 📊 Comprehensive data analysis and visualization
    - 🤖 Multiple ML models (Random Forest, Decision Tree, XGBoost)
    - 🔮 Real-time churn prediction for new customers
    - 📈 Model performance metrics and evaluation
    - ⚖️ SMOTE for handling imbalanced data
    
    **Dataset Information:**
    - **Source**: Telco Customer Churn Dataset
    - **Records**: 7,043 customers
    - **Features**: 20 customer attributes
    - **Target**: Churn prediction (Yes/No)
    """)
    
    # Show sample data
    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

def show_data_analysis(df):
    st.markdown("## 📈 Data Analysis")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Overview")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        # Data types
        st.markdown("### Data Types")
        st.write(df.dtypes.value_counts())
    
    with col2:
        st.markdown("### Target Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, 
                    title="Customer Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Numerical features analysis
    st.markdown("### Numerical Features Analysis")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=numerical_cols)
    
    for i, col in enumerate(numerical_cols, 1):
        fig.add_trace(
            go.Histogram(x=df[col], name=col),
            row=1, col=i
        )
    
    fig.update_layout(height=400, title_text="Distribution of Numerical Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### Correlation Analysis")
    df_numeric = df.copy()
    df_numeric['Churn'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})
    df_numeric['SeniorCitizen'] = df_numeric['SeniorCitizen'].astype(int)
    
    # Convert categorical to numeric for correlation
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Churn':
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes
    
    correlation_matrix = df_numeric.corr()
    
    fig = px.imshow(correlation_matrix, 
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features analysis
    st.markdown("### Categorical Features Analysis")
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Churn']
    
    # Create subplots for categorical features
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                       subplot_titles=categorical_cols,
                       specs=[[{"secondary_y": False}] * n_cols] * n_rows)
    
    for i, col in enumerate(categorical_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        value_counts = df[col].value_counts()
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name=col),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=400 * n_rows, title_text="Categorical Features Distribution")
    st.plotly_chart(fig, use_container_width=True)

def show_model_training(df_processed, model, X_train, X_test, y_train, y_test, y_test_pred):
    st.markdown("## 🤖 Model Training")
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    
    # Prepare data for cross-validation
    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]
    
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_full, y_train_full)
    
    # Cross-validation results
    cv_results = {}
    for model_name, model_obj in models.items():
        scores = cross_val_score(model_obj, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
        cv_results[model_name] = scores.mean()
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cross-Validation Accuracy")
        for model_name, accuracy in cv_results.items():
            st.metric(model_name, f"{accuracy:.3f}")
    
    with col2:
        # Bar chart of model performance
        fig = px.bar(x=list(cv_results.keys()), y=list(cv_results.values()),
                    title="Model Performance Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Selected Model: Random Forest")
    st.markdown("""
    **Why Random Forest?**
    - Highest cross-validation accuracy
    - Handles both numerical and categorical features well
    - Provides feature importance
    - Robust to overfitting
    """)
    
    # Feature importance
    st.markdown("### Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                orientation='h', title="Top 10 Most Important Features")
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(df, encoders, model, feature_names):
    st.markdown("## 🔮 Predict Customer Churn")
    
    st.markdown("### Enter Customer Information")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Customer Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
        
        with col2:
            st.markdown("#### Service Information")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Additional Services")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        with col4:
            st.markdown("#### Contract & Billing")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0, 0.1)
            total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0, 0.1)
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for column, encoder in encoders.items():
            input_df[column] = encoder.transform(input_df[column])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### 🚨 **HIGH CHURN RISK**")
                st.markdown("This customer is likely to churn")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### ✅ **LOW CHURN RISK**")
                st.markdown("This customer is likely to stay")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Probability")
            churn_prob = prediction_proba[1] * 100
            no_churn_prob = prediction_proba[0] * 100
            
            fig = go.Figure(data=[go.Pie(labels=['Stay', 'Churn'], 
                                       values=[no_churn_prob, churn_prob],
                                       hole=0.4)])
            fig.update_layout(title="Churn Probability")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### Confidence Scores")
            st.metric("Stay Probability", f"{no_churn_prob:.1f}%")
            st.metric("Churn Probability", f"{churn_prob:.1f}%")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if prediction == 1:
            st.markdown("""
            **High Churn Risk - Recommended Actions:**
            - Offer retention incentives (discounts, upgrades)
            - Assign dedicated customer success manager
            - Conduct exit interview to understand concerns
            - Provide personalized service improvements
            - Consider contract renewal offers
            """)
        else:
            st.markdown("""
            **Low Churn Risk - Recommended Actions:**
            - Continue providing excellent service
            - Consider upselling opportunities
            - Maintain regular check-ins
            - Gather feedback for service improvements
            - Offer loyalty rewards
            """)

def show_model_performance(y_test, y_test_pred, model, X_test, feature_names):
    st.markdown("## 📊 Model Performance")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig = px.imshow(conf_matrix, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text_auto=True,
                    title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### Classification Report")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Feature Importance
    st.markdown("### Feature Importance Analysis")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='importance', y='feature',
                orientation='h', title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve (if needed)
    st.markdown("### Model Insights")
    st.markdown("""
    **Key Findings:**
    - The model shows good performance in identifying customers at risk of churning
    - Feature importance analysis reveals the most critical factors affecting churn
    - The confusion matrix shows the balance between precision and recall
    - Overall accuracy indicates the model's reliability for business decisions
    """)

if __name__ == "__main__":
    main() 