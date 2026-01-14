#!/usr/bin/env python3
"""
NTU Assignment - Linear Regression + Classification

Exercise 1:
- Linear Regression on California Housing dataset
- Evaluate on test set
- Save plots

Exercise 2:
- Logistic Regression vs KNN on Breast Cancer dataset
- Compare performance on test set
- Save plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)


def banner(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def exercise_1_linear_regression(output_dir: Path, show_plots: bool) -> None:
    banner("EXERCISE 1: LINEAR REGRESSION - CALIFORNIA HOUSING")

    # California Housing requires an internet download the first time.
try:
    housing = fetch_california_housing()
except Exception as e:
    print("\n[ERROR] Could not fetch California Housing dataset.")
    print("This often happens when there is no internet access in the grading environment.")
    print("Error details:", repr(e))
    print("Tip: Re-run once you have internet, or run in an environment where sklearn datasets can download.")
    return

X = housing.data
y = housing.target


print(f"\nDataset shape: {X.shape}")
print(f"Features: {feature_names}")
print("Target variable: Median house value (in $100,000s)")

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")
print(f"Number of features: {len(model.coef_)}")

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mse_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = float(np.sqrt(mse_train))
    r2_train = r2_score(y_train, y_pred_train)

    banner("EVALUATION METRICS ON TEST SET")
    print("Test Set Performance:")
    print(f"  R² Score: {r2_test:.4f}")
    print(f"  Mean Squared Error (MSE): {mse_test:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")

    print("\nTraining Set Performance:")
    print(f"  R² Score: {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")

    banner("FEATURE IMPORTANCE (Coefficients)")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"  {feature:20s}: {coef:10.6f}")
    print(f"  {'Intercept':20s}: {model.intercept_:10.6f}")

    # Plots: Actual vs Predicted and Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual House Prices")
    axes[0].set_ylabel("Predicted House Prices")
    axes[0].set_title(f"Linear Regression: Actual vs Predicted (R²={r2_test:.4f})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Residuals
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted House Prices")
    axes[1].set_ylabel("Residuals (Actual - Predicted)")
    axes[1].set_title("Residual Plot")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    out_path = output_dir / "exercise1_linear_regression.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved plot: {out_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    banner("INTERPRETATION")
    print(f"The model explains {r2_test * 100:.2f}% of the variance in house prices.")
    print(
        f"On average, predictions are off by about ${rmse_test * 100:.2f}k "
        f"(~${rmse_test * 100000:.0f}) per house."
    )


def exercise_2_classification(output_dir: Path, show_plots: bool, k: int) -> None:
    banner("EXERCISE 2: CLASSIFICATION - BREAST CANCER")

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

print(f"\nDataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {list(cancer.target_names)}")
print("  0 = Malignant (cancerous)")
print("  1 = Benign (non-cancerous)")
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", {cancer.target_names[int(u)]: int(c) for u, c in zip(unique, counts)})

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    banner("TRAINING MODELS")

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_clf = LogisticRegression(max_iter=10000, random_state=42)
    lr_clf.fit(X_train_scaled, y_train)
    print("✓ Logistic Regression trained")

    # KNN
    print(f"Training KNN (k={k})...")
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train_scaled, y_train)
    print("✓ KNN trained")

    # Predictions
    y_pred_lr = lr_clf.predict(X_test_scaled)
    y_proba_lr = lr_clf.predict_proba(X_test_scaled)[:, 1]

    y_pred_knn = knn_clf.predict(X_test_scaled)
    y_proba_knn = knn_clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics helper
    def metrics_block(name: str, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)

        banner(f"{name} - TEST SET PERFORMANCE")
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")

        return acc, prec, rec, f1, auc, cm

    lr_scores = metrics_block("LOGISTIC REGRESSION", y_test, y_pred_lr, y_proba_lr)
    knn_scores = metrics_block(f"KNN (k={k})", y_test, y_pred_knn, y_proba_knn)

    # Comparison table
    (lr_acc, lr_prec, lr_rec, lr_f1, lr_auc, cm_lr) = lr_scores
    (knn_acc, knn_prec, knn_rec, knn_f1, knn_auc, cm_knn) = knn_scores

    banner("MODEL COMPARISON")
    print(f"{'Metric':<15} {'Logistic Reg':<15} {'KNN':<15} {'Winner':<15}")
    print("-" * 60)

    def winner(a, b):
        if b > a:
            return "KNN"
        if a > b:
            return "Logistic Reg"
        return "Tie"

    print(f"{'Accuracy':<15} {lr_acc:<15.4f} {knn_acc:<15.4f} {winner(lr_acc, knn_acc):<15}")
    print(f"{'Precision':<15} {lr_prec:<15.4f} {knn_prec:<15.4f} {winner(lr_prec, knn_prec):<15}")
    print(f"{'Recall':<15} {lr_rec:<15.4f} {knn_rec:<15.4f} {winner(lr_rec, knn_rec):<15}")
    print(f"{'F1-Score':<15} {lr_f1:<15.4f} {knn_f1:<15.4f} {winner(lr_f1, knn_f1):<15}")
    print(f"{'ROC-AUC':<15} {lr_auc:<15.4f} {knn_auc:<15.4f} {winner(lr_auc, knn_auc):<15}")

    # Plots: Confusion matrices + ROC curves
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ConfusionMatrixDisplay(cm_lr, display_labels=["Malignant", "Benign"]).plot(ax=ax1, values_format="d")
    ax1.set_title("Logistic Regression - Confusion Matrix")

    ax2 = fig.add_subplot(gs[0, 1])
    ConfusionMatrixDisplay(cm_knn, display_labels=["Malignant", "Benign"]).plot(ax=ax2, values_format="d")
    ax2.set_title(f"KNN (k={k}) - Confusion Matrix")

    ax3 = fig.add_subplot(gs[1, :])
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

    ax3.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={lr_auc:.4f})", lw=2)
    ax3.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={knn_auc:.4f})", lw=2)
    ax3.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves Comparison")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="lower right")

    plt.tight_layout()
    out_path = output_dir / "exercise2_classification.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved plot: {out_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    banner("CONCLUSION")
    better_model = "Logistic Regression" if lr_acc >= knn_acc else "KNN"
    print(f"Best Model (by Accuracy): {better_model}")
    print(f"Accuracy Difference: {abs(lr_acc - knn_acc) * 100:.2f}%")
    print("\nNote:")
    print("- Logistic Regression is often strong on this dataset after scaling (linear separability).")
    print("- KNN is sensitive to k and scaling; try tuning k for potentially better results.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NTU Assignment script (Regression + Classification)")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save generated plots (default: outputs)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively (default: save only)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for KNN (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_1_linear_regression(output_dir=output_dir, show_plots=args.show_plots)
    exercise_2_classification(output_dir=output_dir, show_plots=args.show_plots, k=args.k)

    print("\nAll done ✅")
    print(f"Plots saved in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
