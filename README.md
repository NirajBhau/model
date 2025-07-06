# Customer Churn Prediction - Streamlit App

A comprehensive machine learning application for predicting customer churn in telecommunications using Streamlit.

## 🚀 Features

### 📊 Data Analysis
- Interactive data exploration and visualization
- Statistical analysis of customer demographics
- Feature correlation analysis
- Distribution plots for numerical and categorical features

### 🤖 Machine Learning
- Multiple ML models (Random Forest, Decision Tree, XGBoost)
- SMOTE for handling imbalanced data
- Cross-validation for model comparison
- Feature importance analysis

### 🔮 Prediction Interface
- User-friendly form for customer data input
- Real-time churn probability prediction
- Confidence scores and recommendations
- Visual probability charts

### 📈 Model Performance
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Classification report
- Model insights and recommendations

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── V_Customer_Churn_Prediction_using_ML.ipynb  # Original analysis notebook
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

## 📊 Dataset Information

The application uses the **Telco Customer Churn Dataset** with the following characteristics:

- **Records**: 7,043 customers
- **Features**: 20 customer attributes
- **Target**: Churn prediction (Yes/No)
- **Churn Rate**: ~26.5%

### Key Features:
- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Services**: Phone Service, Internet Service, Multiple Lines
- **Add-ons**: Online Security, Online Backup, Device Protection, Tech Support
- **Streaming**: Streaming TV, Streaming Movies
- **Contract**: Contract type, Paperless Billing, Payment Method
- **Billing**: Tenure, Monthly Charges, Total Charges

## 🎯 How to Use

### 1. Home Page
- Overview of the project and dataset statistics
- Quick insights into customer distribution

### 2. Data Analysis
- Explore customer demographics and service usage
- View feature correlations and distributions
- Understand data patterns and relationships

### 3. Model Training
- Compare different ML algorithms
- View cross-validation results
- Analyze feature importance

### 4. Predict Churn
- Enter customer information using the form
- Get real-time churn predictions
- View probability scores and recommendations

### 5. Model Performance
- Review model evaluation metrics
- Analyze confusion matrix
- Understand model strengths and limitations

## 🔧 Technical Details

### Models Used:
- **Random Forest**: Primary model (best performance)
- **Decision Tree**: Baseline comparison
- **XGBoost**: Advanced gradient boosting

### Data Preprocessing:
- Label encoding for categorical variables
- SMOTE for handling class imbalance
- Feature scaling and normalization

### Performance Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation scores

## 📈 Business Impact

This application helps businesses:

1. **Identify At-Risk Customers**: Predict which customers are likely to churn
2. **Reduce Churn Rate**: Take proactive measures to retain customers
3. **Optimize Resources**: Focus retention efforts on high-risk customers
4. **Improve Customer Experience**: Understand factors driving churn
5. **Data-Driven Decisions**: Make informed business strategies

## 🎨 Features Highlights

- **Interactive Dashboard**: Beautiful and intuitive user interface
- **Real-time Predictions**: Instant churn probability calculations
- **Comprehensive Analysis**: Detailed data exploration and visualization
- **Business Recommendations**: Actionable insights for customer retention
- **Responsive Design**: Works on desktop and mobile devices

## 🔮 Prediction Example

When you input customer data, the app provides:

- **Churn Risk Level**: High/Low risk classification
- **Probability Scores**: Percentage likelihood of churning
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Recommendations**: Specific actions to take based on risk level

## 📊 Model Performance

The Random Forest model achieves:
- **Accuracy**: ~78%
- **Precision**: ~58% (for churn class)
- **Recall**: ~59% (for churn class)
- **F1-Score**: ~58% (for churn class)

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Enhancing the model performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Telco Customer Churn Dataset
- Libraries: Streamlit, Scikit-learn, Plotly, Pandas
- Community: Open source contributors and data science community

---

**Happy Predicting! 🎯** 