"""
Telco Data ML Model Analysis
Builds and compares three classification models:
- Logistic Regression
- Decision Tree
- Naive Bayes

Generates ROC curves, confusion matrices, and model comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    roc_auc_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class TelcoModelAnalysis:
    """Class to handle telco data analysis and model building"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        """Load telco data from CSV"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData info:\n{self.df.info()}")
        return self.df
    
    def preprocess_data(self, target_column=None):
        """Preprocess data: handle missing values, encode categorical variables"""
        print("\n" + "="*50)
        print("PREPROCESSING DATA")
        print("="*50)
        
        # Make a copy to avoid modifying original
        df = self.df.copy()
        
        # Handle missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        df = df.dropna()
        
        # Identify categorical and numeric columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nCategorical columns: {categorical_cols}")
        print(f"Numeric columns: {numeric_cols}")
        
        # Find target column (usually last object column or specified)
        if target_column is None:
            target_column = categorical_cols[-1] if categorical_cols else None
        
        if target_column is None:
            raise ValueError("Could not identify target column. Please specify it.")
        
        print(f"\nTarget column: {target_column}")
        print(f"Target value counts:\n{df[target_column].value_counts()}")
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Update categorical and numeric columns (excluding target)
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        # Encode categorical variables
        self.le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.le_dict[col] = le
        
        # Encode target variable
        self.le_target = LabelEncoder()
        y = self.le_target.fit_transform(y)
        
        print(f"\nTarget encoding: {dict(zip(self.le_target.classes_, self.le_target.transform(self.le_target.classes_)))}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Class distribution in training set:\n{pd.Series(self.y_train).value_counts()}")
        
    def build_models(self):
        """Build all three models"""
        print("\n" + "="*50)
        print("BUILDING MODELS")
        print("="*50)
        
        # Logistic Regression
        print("\n1. Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['Logistic Regression'].fit(self.X_train, self.y_train)
        print("✓ Logistic Regression trained")
        
        # Decision Tree
        print("\n2. Training Decision Tree...")
        self.models['Decision Tree'] = DecisionTreeClassifier(random_state=42, max_depth=10)
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        print("✓ Decision Tree trained")
        
        # Naive Bayes
        print("\n3. Training Naive Bayes...")
        self.models['Naive Bayes'] = GaussianNB()
        self.models['Naive Bayes'].fit(self.X_train, self.y_train)
        print("✓ Naive Bayes trained")
        
    def evaluate_models(self):
        """Evaluate all models and generate predictions"""
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\nGenerating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.predictions[model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[idx],
                cbar=False,
                square=True
            )
            axes[idx].set_title(model_name, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved as 'confusion_matrices.png'")
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("Generating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name in self.models.keys():
            y_pred_proba = self.predictions[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, 
                tpr, 
                linewidth=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})'
            )
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved as 'roc_curves.png'")
        plt.close()
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        print("Generating model comparison...")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.metrics).T
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = comparison_df[metric].values
            model_names = comparison_df.index
            
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{height:.4f}',
                    ha='center', 
                    va='bottom',
                    fontweight='bold'
                )
            
            ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison saved as 'model_comparison.png'")
        plt.close()
    
    def plot_roc_auc_comparison(self):
        """Plot ROC-AUC scores comparison"""
        print("Generating ROC-AUC comparison...")
        
        roc_auc_scores = {model: self.metrics[model]['roc_auc'] for model in self.models.keys()}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.barh(list(roc_auc_scores.keys()), list(roc_auc_scores.values()), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left',
                va='center',
                fontweight='bold',
                fontsize=11
            )
        
        ax.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ ROC-AUC comparison saved as 'roc_auc_comparison.png'")
        plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report"""
        print("\n" + "="*50)
        print("MODEL SUMMARY REPORT")
        print("="*50)
        
        summary_df = pd.DataFrame(self.metrics).T
        print("\n" + summary_df.to_string())
        
        # Best model
        best_model = summary_df['roc_auc'].idxmax()
        print(f"\n✓ Best Model (by ROC-AUC): {best_model} ({summary_df.loc[best_model, 'roc_auc']:.4f})")
        
        # Save summary to CSV
        summary_df.to_csv('model_summary.csv')
        print("✓ Summary saved to 'model_summary.csv'")
        
        return summary_df
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print("TELCO DATA ML MODEL ANALYSIS")
        print("="*70)
        
        self.load_data()
        self.preprocess_data()
        self.build_models()
        self.evaluate_models()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_model_comparison()
        self.plot_roc_auc_comparison()
        summary = self.generate_summary_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  1. confusion_matrices.png - Confusion matrices for all models")
        print("  2. roc_curves.png - ROC curves comparison")
        print("  3. model_comparison.png - Performance metrics comparison")
        print("  4. roc_auc_comparison.png - ROC-AUC scores comparison")
        print("  5. model_summary.csv - Summary metrics in CSV format")
        
        return summary


if __name__ == "__main__":
    # Update this path to your telco data
    DATA_PATH = "processed_telco_data (2) (2).csv"
    
    # Create analyzer and run full analysis
    analyzer = TelcoModelAnalysis(DATA_PATH)
    summary = analyzer.run_full_analysis()
