# Telco Data ML Model Analysis

## Overview
This project builds and compares three machine learning classification models on telco customer churn data:
- **Logistic Regression**
- **Decision Tree**
- **Naive Bayes**

The analysis generates comprehensive visualizations including ROC curves, confusion matrices, and model performance comparisons.

## Features
✅ Automated data loading and preprocessing  
✅ Categorical variable encoding and feature scaling  
✅ Three classification models with hyperparameter tuning  
✅ Comprehensive model evaluation metrics  
✅ ROC curves comparison  
✅ Confusion matrices for all models  
✅ Performance metrics visualization  
✅ Summary report in CSV format  

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/devbagaletamang-rgb/Business-Intelligence-Assignment.git
cd Business-Intelligence-Assignment

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from telco_model_analysis import TelcoModelAnalysis

# Initialize with your data path
analyzer = TelcoModelAnalysis("processed_telco_data (2) (2).csv")

# Run complete analysis
summary = analyzer.run_full_analysis()
```

### Command Line
```bash
python telco_model_analysis.py
```

**Note:** Update the `DATA_PATH` variable in `__main__` section to point to your telco data CSV file.

## Output Files

The script generates the following visualizations and reports:

1. **confusion_matrices.png** - Side-by-side confusion matrices for all three models
2. **roc_curves.png** - ROC curve comparison with AUC scores
3. **model_comparison.png** - Performance metrics comparison (Accuracy, Precision, Recall, F1-Score)
4. **roc_auc_comparison.png** - ROC-AUC scores bar chart
5. **model_summary.csv** - Summary metrics in tabular format

## Models Description

### 1. Logistic Regression
- Linear classification model
- Best for interpretability
- Output: Probability scores for binary classification

### 2. Decision Tree
- Tree-based model with max_depth=10
- Good for feature importance analysis
- Handles non-linear relationships

### 3. Naive Bayes
- Probabilistic model based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction

## Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

## Data Preprocessing

The script automatically:
1. Loads CSV data
2. Handles missing values (drops rows with NaN)
3. Identifies target variable (last categorical column)
4. Encodes categorical variables using LabelEncoder
5. Splits data: 80% training, 20% testing (stratified)
6. Scales features using StandardScaler

## Class Structure

### TelcoModelAnalysis
Main class with the following methods:

- `load_data()`: Loads telco CSV data
- `preprocess_data(target_column=None)`: Preprocesses and prepares data
- `build_models()`: Trains all three models
- `evaluate_models()`: Evaluates models and calculates metrics
- `plot_confusion_matrices()`: Creates confusion matrix visualization
- `plot_roc_curves()`: Creates ROC curve comparison
- `plot_model_comparison()`: Creates performance metrics comparison
- `plot_roc_auc_comparison()`: Creates ROC-AUC bar chart
- `generate_summary_report()`: Generates and saves summary report
- `run_full_analysis()`: Runs complete analysis pipeline

## Example Output

```
===========================================================
TELCO DATA ML MODEL ANALYSIS
===========================================================

Loading data...
Data shape: (7043, 21)

==================================================
PREPROCESSING DATA
==================================================

Target column: Churn
Target value counts:
No     5174
Yes    1869
...

==================================================
BUILDING MODELS
==================================================

1. Training Logistic Regression...
✓ Logistic Regression trained

2. Training Decision Tree...
✓ Decision Tree trained

3. Training Naive Bayes...
✓ Naive Bayes trained

==================================================
EVALUATING MODELS
==================================================

Logistic Regression:
Accuracy:  0.8050
Precision: 0.6428
Recall:    0.5645
F1-Score:  0.6012
ROC-AUC:   0.8621

Decision Tree:
Accuracy:  0.7420
Precision: 0.5947
Recall:    0.6180
F1-Score:  0.6062
ROC-AUC:   0.7589

Naive Bayes:
Accuracy:  0.7850
Precision: 0.5890
Recall:    0.6789
F1-Score:  0.6303
ROC-AUC:   0.8512

==================================================
MODEL SUMMARY REPORT
==================================================

✓ Best Model (by ROC-AUC): Logistic Regression (0.8621)

ANALYSIS COMPLETE!

Generated files:
  1. confusion_matrices.png - Confusion matrices for all models
  2. roc_curves.png - ROC curves comparison
  3. model_comparison.png - Performance metrics comparison
  4. roc_auc_comparison.png - ROC-AUC scores comparison
  5. model_summary.csv - Summary metrics in CSV format
```

## Requirements

See `requirements.txt` for complete dependencies:
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0
- matplotlib>=3.4.0
- seaborn>=0.11.0

## Author
Dev Bagale Tamang

## License
MIT License

## Contributing
Feel free to submit issues and enhancement requests!

## Repository
[Business-Intelligence-Assignment](https://github.com/devbagaletamang-rgb/Business-Intelligence-Assignment)
