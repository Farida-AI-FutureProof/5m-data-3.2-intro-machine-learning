#!/usr/bin/env python3
"""
NTU Assignment - Module 3.2 Intro to Machine Learning

Exercise 1:
- Linear Regression on California Housing (fallback to load_diabetes if blocked)
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

from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
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


def load_regression_dataset():
    """
    Try California Housing; if blocked (HTTP 403 or no internet), fall back to load_diabetes.
    Returns: X, y, feature_names, dataset_name
    """
    try:
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target
        feature_names = housing.feature_names
        dataset_name = "California Housing"
        return X, y, feature_names, dataset_name
    except Exception as e:
        print("\n[WARN] Could not fetch California Housing dataset (often HTTP 403 / blocked).")
        print("Falling back to sklearn load_diabetes().")
        print("Error details:", repr(e))

        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        feature_names = list(diabetes.feature_names)
        dataset_name = "Diabetes (fallback)"
        return X, y, feature_names, dataset_name


def exercise_1_linear_regression(output_dir: Path, show_plots: bool) -> None:
    banner("EXERCISE 1: LINEAR REGRESSION")

    X, y, feature_names, dataset_name = load_regression_dataset()

    print(f"\nDataset: {dataset_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print("Target variable: numeric target (housing: $100,000s; diabetes: disease progression)")

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

    banner("EVALUATION METRICS (TEST SET)")
    print("Test Set Performance:")
    print(f"  R² Score: {r2_test:.4f}")
    print(f"  Mean Squared Error (MSE): {mse_test:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")

    print("\nTraining Set Performance:")
    print(f"  R² Score: {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")

    banner("FEATURE COEFFICIENTS")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"  {str(feature):20s}: {coef:12.6f}")
    print(f"  {'Intercept':20s}: {model.intercept_:12.6f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
    axes[0].plot(
        [np.min(y_test), np.max(y_test)],
        [np.min(y_test), np.max(y_test)],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Linear Regression ({dataset_name})\nR² = {r2_test:.4f}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Residuals
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted")
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
    print(f"The model explains {r2_test * 100:.2f}% of the variance in the target.")
    print(f"RMSE (scale depends on dataset): {rmse_test:.4f}")


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

    # Scale features
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

    def report_metrics(name: str, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)

        banner(f"{name} - TEST SET PERFORMANCE")
        print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")

        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")

        return acc, prec, rec, f1, auc, cm

    lr_acc, lr_prec, lr_rec, lr_f1, lr_auc, cm_lr = report_metrics(
        "LOGISTIC REGRESSION", y_test, y_pred_lr, y_proba_lr
    )
    knn_acc, knn_prec, knn_rec, knn_f1, knn_auc, cm_knn = report_metrics(
        f"KNN (k={k})", y_test, y_pred_knn, y_proba_knn
    )

    banner("MODEL COMPARISON")

    def winner(a: float, b: float) -> str:
        if b > a:
            return "KNN"
        if a > b:
            return "Logistic Reg"
        return "Tie"

    print(f"{'Metric':<15} {'Logistic Reg':<15} {'KNN':<15} {'Winner':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {lr_acc:<15.4f} {knn_acc:<15.4f} {winner(lr_acc, knn_acc):<15}")
    print(f"{'Precision':<15} {lr_prec:<15.4f} {knn_prec:<15.4f} {winner(lr_prec, knn_prec):<15}")
    print(f"{'Recall':<15} {lr_rec:<15.4f} {knn_rec:<15.4f} {winner(lr_rec, knn_rec):<15}")
    print(f"{'F1-Score':<15} {lr_f1:<15.4f} {knn_f1:<15.4f} {winner(lr_f1, knn_f1):<15}")
    print(f"{'ROC-AUC':<15} {lr_auc:<15.4f} {knn_auc:<15.4f} {winner(lr_auc, knn_auc):<15}")

    # Plots: confusion matrices + ROC curves
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
    better = "Logistic Regression" if lr_acc >= knn_acc else "KNN"
    print(f"Best Model (by Accuracy): {better}")
    print(f"Accuracy Difference: {abs(lr_acc - knn_acc) * 100:.2f}%")
    print("\nNotes:")
    print("- Logistic Regression often performs very well here after scaling.")
    print("- KNN performance can change with k; try tuning k for improvement.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NTU Assignment: Regression + Classification")
    parser.add_argument("--output-dir", default="outputs", help="Folder to save plots (default: outputs)")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    parser.add_argument("--k", type=int, default=5, help="k for KNN (default: 5)")
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
