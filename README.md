# Telco Customer Churn Prediction Model

## Project Overview

This project develops a machine learning model to predict customer churn rate for a telecom company using the **Telco Customer Churn dataset from Kaggle**.

### Dataset Information
- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Records**: ~7,000 customers
- **Features**: 21 attributes including demographics, services, and account information
- **Target Variable**: Churn (Yes/No)

## Project Structure

```
├── telco_churn_model.py          # Main model training script
├── telco_churn_eda.py            # Exploratory Data Analysis
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── model_comparison.png          # Output visualization
```

## Features

### Data Attributes
- **Demographic Info**: Gender, Age Group, Senior Citizen Status
- **Account Information**: Tenure, Contract Type, Payment Method
- **Services**: Phone, Internet, Backup, Device Protection, Tech Support
- **Charges**: Monthly Charges, Total Charges

### Target Variable
- **Churn**: Whether the customer left the company (Yes/No)

## Models Implemented

### 1. Logistic Regression
- **Advantages**: Fast, interpretable, good baseline
- **Use Case**: Quick predictions, understanding feature importance

### 2. Random Forest
- **Advantages**: Handles non-linearity, feature importance ranking
- **Use Case**: Balanced approach between interpretability and performance

### 3. Gradient Boosting
- **Advantages**: High accuracy, handles complex patterns
- **Use Case**: Best performance for prediction tasks

## Data Preprocessing

1. **Handling Missing Values**: Filled with appropriate methods
2. **Encoding Categorical Variables**: Label Encoding for all categories
3. **Feature Scaling**: StandardScaler for numerical features
4. **Train-Test Split**: 80-20 split with stratification

## Model Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Visual representation of predictions

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Download Dataset
1. Go to [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in the project directory

### Run the Model
```bash
python telco_churn_model.py
```

### Run EDA
```bash
python telco_churn_eda.py
```

## Expected Results

Typical performance metrics on test set:
- **Best Model (Gradient Boosting)**:
  - Accuracy: ~80-82%
  - ROC-AUC: ~85-87%
  - F1-Score: ~60-65%
  - Recall: ~55-60%

## Key Insights

### High-Risk Churn Factors
1. Month-to-Month contracts have 3x higher churn
2. Customers with no internet service churn less
3. High monthly charges correlate with churn
4. Fiber optic customers churn more

### Retention Strategies
1. Encourage longer-term contracts
2. Bundle services to increase stickiness
3. Target high-value customers with retention offers
4. Improve service quality for fiber optic users

## Future Enhancements

1. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
2. **Feature Engineering**: Create new features (e.g., charges per month ratios)
3. **Class Imbalance**: Apply SMOTE or class weights
4. **Ensemble Methods**: Stack multiple models for better predictions
5. **Deployment**: Flask/FastAPI for real-time predictions
6. **Interpretability**: SHAP values for feature importance

## File Descriptions

### telco_churn_model.py
Main script containing the `TelcoChurnPredictor` class with methods for:
- Data loading and preprocessing
- Model training (3 different algorithms)
- Model evaluation and comparison
- Results visualization

### telco_churn_eda.py
Exploratory Data Analysis script with:
- Univariate analysis
- Bivariate analysis
- Correlation heatmap
- Churn distribution analysis
- Feature relationships

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

This project is for educational purposes.

## Contact & Support

For questions or improvements, please create an issue or pull request.

---

**Last Updated**: May 2026
**Model Version**: 1.0