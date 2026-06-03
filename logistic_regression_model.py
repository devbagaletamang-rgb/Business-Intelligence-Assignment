"""
Logistic Regression Model
"""

from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    """Logistic Regression Model"""
    
    def __init__(self, random_state=42, max_iter=1000):
        """Initialize Logistic Regression"""
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nTraining Logistic Regression...")
        self.model.fit(X_train, y_train)
        print("✓ Logistic Regression trained successfully")
        
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        return self.model.predict_proba(X_test)
    
    def get_model(self):
        """Return the model"""
        return self.model
