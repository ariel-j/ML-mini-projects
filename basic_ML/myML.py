import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_polynomial=True):
        self.add_polynomial = add_polynomial
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Create DataFrame for easier manipulation
        X_transformed = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 
                                               'petal_length', 'petal_width'])
        
        # Add ratio features
        X_transformed['sepal_ratio'] = X_transformed['sepal_length'] / X_transformed['sepal_width']
        X_transformed['petal_ratio'] = X_transformed['petal_length'] / X_transformed['petal_width']
        
        # Add polynomial features for non-linear relationships
        if self.add_polynomial:
            for col in ['sepal_length', 'petal_length']:
                X_transformed[f'{col}_squared'] = X_transformed[col] ** 2
                
        return X_transformed.values

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'feature_engineer__add_polynomial': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Print results
print("\nBest parameters:", grid_search.best_params_)
print("\nCross-validation scores:", cross_val_score(best_model, X_train, y_train, cv=5))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance analysis
def plot_feature_importance(model, feature_names):
    # Get feature importance from the Random Forest classifier
    importance = model.named_steps['classifier'].feature_importances_
    
    # Get transformed feature names
    engineered_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                          'sepal_ratio', 'petal_ratio']
    if model.named_steps['feature_engineer'].add_polynomial:
        engineered_features.extend(['sepal_length_squared', 'petal_length_squared'])
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': engineered_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

# Plot feature importance
plot_feature_importance(best_model, iris.feature_names)

# Learning curves
def plot_learning_curves(model, X, y):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Plot learning curves
from sklearn.model_selection import learning_curve
plot_learning_curves(best_model, X, y)