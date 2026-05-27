"""
Telco Customer Churn Prediction Model
This script builds a machine learning model to predict customer churn for a telecom company.
Dataset: Kaggle Telco Customer Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

class TelcoChurnPredictor:
    """Class to handle telco customer churn prediction"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load the telco customer churn dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("\n" + "="*50)
        print("PREPROCESSING DATA")
        print("="*50)
        
        # Create a copy
        df = self.df.copy()
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"\nCategorical columns: {categorical_cols}")
        
        # Encode target variable (Churn)
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Remove customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Encode other categorical variables
        for col in categorical_cols:
            if col != 'Churn':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Convert TotalCharges to numeric (handle empty strings)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Convert tenure to numeric if needed
        if 'tenure' in df.columns:
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        
        # Convert MonthlyCharges to numeric
        if 'MonthlyCharges' in df.columns:
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        
        print(f"\nProcessed data shape: {df.shape}")
        print(f"Data types after preprocessing:\n{df.dtypes}")
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "="*50)
        print("SPLITTING DATA")
        print("="*50)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        print(f"\nTarget variable distribution:\n{y.value_counts()}")
        print(f"Churn rate: {y.sum() / len(y) * 100:.2f}%")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*50)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        
        return model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "="*50)
        print("TRAINING GRADIENT BOOSTING")
        print("="*50)
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        
        return model
    
    def evaluate_model(self, model, model_name="Model"):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print(f"EVALUATING {model_name}")
        print("="*50)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nAccuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
        
        return y_pred, y_pred_proba
    
    def plot_results(self, models_dict):
        """Plot comparison of all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Telco Customer Churn Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'ROC-AUC': []}
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            results['Model'].append(model_name)
            results['Accuracy'].append(accuracy_score(self.y_test, y_pred))
            results['Precision'].append(precision_score(self.y_test, y_pred))
            results['Recall'].append(recall_score(self.y_test, y_pred))
            results['F1-Score'].append(f1_score(self.y_test, y_pred))
            results['ROC-AUC'].append(roc_auc_score(self.y_test, y_pred_proba))
        
        results_df = pd.DataFrame(results)
        
        # Plot 1: Accuracy Comparison
        axes[0, 0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Precision vs Recall
        axes[0, 1].scatter(results_df['Recall'], results_df['Precision'], s=200)
        for i, model in enumerate(results_df['Model']):
            axes[0, 1].annotate(model, (results_df['Recall'][i], results_df['Precision'][i]))
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        
        # Plot 3: F1-Score Comparison
        axes[1, 0].bar(results_df['Model'], results_df['F1-Score'], color='lightcoral')
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: ROC-AUC Comparison
        axes[1, 1].bar(results_df['Model'], results_df['ROC-AUC'], color='lightgreen')
        axes[1, 1].set_title('ROC-AUC Comparison')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].set_ylim([0, 1])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nModel comparison plot saved as 'model_comparison.png'")
        plt.show()
        
        return results_df

def main():
    """Main execution function"""
    # Initialize predictor
    predictor = TelcoChurnPredictor('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Load and preprocess data
    predictor.load_data()
    df_processed = predictor.preprocess_data()
    
    # Split data
    predictor.split_data(df_processed)
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Logistic Regression': predictor.train_logistic_regression(),
        'Random Forest': predictor.train_random_forest(),
        'Gradient Boosting': predictor.train_gradient_boosting()
    }
    
    # Evaluate all models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    for name, model in models.items():
        predictor.evaluate_model(model, name)
    
    # Plot results
    results_df = predictor.plot_results(models)
    print("\nModel Comparison Results:")
    print(results_df)
    
    # Return best model
    best_model_idx = results_df['ROC-AUC'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    print(f"\nBest Model: {best_model_name}")
    
    return models, results_df

if __name__ == "__main__":
    models, results = main()