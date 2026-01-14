# Assignment

## Instructions

Complete the following exercises using Python.

1. Linear Regression Exercise:
   Using the California Housing dataset from scikit-learn, create a linear regression model to predict house prices.
   Evaluate the performance of Linear Regression on test set.

   ```python
   from sklearn.datasets import fetch_california_housing

   # Load dataset
   housing = fetch_california_housing()
   ```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("="*70)
print("EXERCISE 1: LINEAR REGRESSION - CALIFORNIA HOUSING")
print("="*70)

# Load dataset
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

print(f"\nDataset shape: {X_housing.shape}")
print(f"Features: {housing.feature_names}")
print(f"Target variable: House prices (in hundreds of thousands of dollars)")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")
print(f"Number of features: {len(lr_model.coef_)}")

# Make predictions on test set
y_pred_test = lr_model.predict(X_test)
y_pred_train = lr_model.predict(X_train)

# Evaluate performance on test set
print("\n" + "="*70)
print("EVALUATION METRICS ON TEST SET")
print("="*70)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print(f"\nTest Set Performance:")
print(f"  R² Score: {r2_test:.4f}")
print(f"  Mean Squared Error (MSE): {mse_test:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")

print(f"\nTraining Set Performance:")
print(f"  R² Score: {r2_train:.4f}")
print(f"  RMSE: {rmse_train:.4f}")

# Feature Coefficients
print(f"\n" + "="*70)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*70)
for feature, coef in zip(housing.feature_names, lr_model.coef_):
    print(f"  {feature:20s}: {coef:10.6f}")
print(f"  {'Intercept':20s}: {lr_model.intercept_:10.6f}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual House Prices', fontsize=11)
axes[0].set_ylabel('Predicted House Prices', fontsize=11)
axes[0].set_title(f'Linear Regression: Actual vs Predicted\nR² = {r2_test:.4f}', fontsize=12)
axes[0].grid(alpha=0.3)
axes[0].legend()

# Plot 2: Residuals
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted House Prices', fontsize=11)
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
axes[1].set_title('Residual Plot', fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"The model explains {r2_test*100:.2f}% of the variance in house prices.")
print(f"On average, predictions are off by ${rmse_test*100:.2f} thousand (${rmse_test*100000:.2f}) per house.")

2. Classification Exercise:
   Using the breast cancer dataset from scikit-learn, build classification models to predict malignant vs benign tumors.
   Compare Logistic Regression and KNN performance on test set.

   ```python
   from sklearn.datasets import load_breast_cancer

   # Load dataset
   cancer = load_breast_cancer()
   ```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve)

print("="*70)
print("EXERCISE 2: CLASSIFICATION - BREAST CANCER")
print("="*70)

# Load dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print(f"\nDataset shape: {X_cancer.shape}")
print(f"Number of features: {X_cancer.shape[1]}")
print(f"Target classes: {cancer.target_names}")
print(f"  0 = Malignant (cancerous)")
print(f"  1 = Benign (non-cancerous)")
print(f"Class distribution: {dict(zip(cancer.target_names, np.bincount(y_cancer)))}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize features (important for both models, especially KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr_classifier = LogisticRegression(max_iter=10000, random_state=42)
lr_classifier.fit(X_train_scaled, y_train)
print("✓ Logistic Regression trained")

# Model 2: KNN
print("Training KNN (k=5)...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)
print("✓ KNN trained")

# Predictions
y_pred_lr = lr_classifier.predict(X_test_scaled)
y_pred_lr_proba = lr_classifier.predict_proba(X_test_scaled)[:, 1]

y_pred_knn = knn_classifier.predict(X_test_scaled)
y_pred_knn_proba = knn_classifier.predict_proba(X_test_scaled)[:, 1]

# Evaluate Logistic Regression
print("\n" + "="*70)
print("LOGISTIC REGRESSION - TEST SET PERFORMANCE")
print("="*70)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_pred_lr_proba)

print(f"\nAccuracy:  {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_roc_auc:.4f}")

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_lr[0,0]}")
print(f"  False Positives: {cm_lr[0,1]}")
print(f"  False Negatives: {cm_lr[1,0]}")
print(f"  True Positives:  {cm_lr[1,1]}")

# Evaluate KNN
print("\n" + "="*70)
print("KNN (k=5) - TEST SET PERFORMANCE")
print("="*70)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)
knn_roc_auc = roc_auc_score(y_test, y_pred_knn_proba)

print(f"\nAccuracy:  {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall:    {knn_recall:.4f}")
print(f"F1-Score:  {knn_f1:.4f}")
print(f"ROC-AUC:   {knn_roc_auc:.4f}")

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_knn[0,0]}")
print(f"  False Positives: {cm_knn[0,1]}")
print(f"  False Negatives: {cm_knn[1,0]}")
print(f"  True Positives:  {cm_knn[1,1]}")

# Model Comparison
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

print(f"\n{'Metric':<15} {'Logistic Reg':<15} {'KNN (k=5)':<15} {'Winner':<15}")
print("-" * 60)
print(f"{'Accuracy':<15} {lr_accuracy:<15.4f} {knn_accuracy:<15.4f} {'KNN' if knn_accuracy > lr_accuracy else 'Logistic Reg':<15}")
print(f"{'Precision':<15} {lr_precision:<15.4f} {knn_precision:<15.4f} {'KNN' if knn_precision > lr_precision else 'Logistic Reg':<15}")
print(f"{'Recall':<15} {lr_recall:<15.4f} {knn_recall:<15.4f} {'KNN' if knn_recall > lr_recall else 'Logistic Reg':<15}")
print(f"{'F1-Score':<15} {lr_f1:<15.4f} {knn_f1:<15.4f} {'KNN' if knn_f1 > lr_f1 else 'Logistic Reg':<15}")
print(f"{'ROC-AUC':<15} {lr_roc_auc:<15.4f} {knn_roc_auc:<15.4f} {'KNN' if knn_roc_auc > lr_roc_auc else 'Logistic Reg':<15}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrices
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'],
            cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label', fontsize=11)
axes[0, 0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'],
            cbar_kws={'label': 'Count'})
axes[0, 1].set_title('KNN (k=5) - Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('True Label', fontsize=11)
axes[0, 1].set_xlabel('Predicted Label', fontsize=11)

# Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc]
knn_scores = [knn_accuracy, knn_precision, knn_recall, knn_f1, knn_roc_auc]

x = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x - width/2, lr_scores, width, label='Logistic Regression', alpha=0.8, color='steelblue')
axes[1, 0].bar(x + width/2, knn_scores, width, label='KNN (k=5)', alpha=0.8, color='orange')
axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_ylim([0.85, 1.05])

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_proba)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn_proba)

axes[1, 1].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_roc_auc:.4f})', lw=2.5, color='steelblue')
axes[1, 1].plot(fpr_knn, tpr_knn, label=f'KNN (AUC={knn_roc_auc:.4f})', lw=2.5, color='orange')
axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
axes[1, 1].set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
better_model = 'Logistic Regression' if lr_accuracy >= knn_accuracy else 'KNN'
print(f"\nBest Model: {better_model}")
print(f"Accuracy Difference: {abs(lr_accuracy - knn_accuracy)*100:.2f}%")
print(f"\nLogistic Regression is generally better for this dataset because:")
print(f"  - The cancer classes are linearly separable in high dimensions")
print(f"  - It's computationally more efficient")
print(f"  - Provides probabilistic predictions")

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
