#!/usr/bin/env python3

"""
MACHINE LEARNING - RENATO VIVAR - Iterative Outlier Removal with Lasso Regression
This script performs iterative outlier removal using Lasso regression.
It visualizes the outlier removal process and evaluates the model's performance before and after outlier
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, learning_curve

def iterative_regression_outliers(x_data, y_data, alpha=3.5, iterations=50, visualize=False):
    train_x = x_data.copy()
    train_y = y_data.copy()

    visualize_steps = [1, 10, 20, 30, 40, 50]
    saved_plots = []

    if visualize:
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 6 plots
        axs = axs.flatten()

    for i in range(iterations):
        model = linear_model.Lasso(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_y)

        y_pred = model.predict(train_x)
        aux_sse = (y_pred - train_y) ** 2
        aux_index = np.argmax(aux_sse)

        if visualize and (i + 1) in visualize_steps:
            idx = visualize_steps.index(i + 1)
            axs[idx].scatter(train_y, y_pred, alpha=0.4, s=10)
            axs[idx].scatter(train_y[aux_index], y_pred[aux_index], color='red', s=30)
            axs[idx].plot([min(train_y), max(train_y)], [min(train_y), max(train_y)], '--', color='black')
            axs[idx].set_title(f'Iteration {i + 1}')
            axs[idx].set_xlabel("Actual")
            axs[idx].set_ylabel("Predicted")

        # Remove outlier
        train_x = np.delete(train_x, aux_index, axis=0)
        train_y = np.delete(train_y, aux_index, axis=0)

    if visualize:
        plt.tight_layout()
        plt.suptitle("Outlier Removal Process (Red = Removed Sample)", y=1.02, fontsize=16)
        plt.savefig("plots_IOR/outlier_removal_process.png")
        plt.close()

    return train_x, train_y

def main():
    os.makedirs("plots_IOR", exist_ok=True)

    # Load data
    x_train = np.load('DATA/X_train.npy')
    y_train = np.load('DATA/y_train.npy')

    feature_names = ['AIR TEMPERATURE', 'WATER TEMP', 'WIND SPEED', 'WIND DIRECTION', 'ILUMINATION']
    dataframeX = pd.DataFrame(x_train, columns=feature_names)
    dataframeY = pd.DataFrame(y_train, columns=['TOXIC ALGUE'])

    # Baseline model (before outlier removal)
    baseline_model = linear_model.Lasso(alpha=3.5, fit_intercept=True)
    baseline_model.fit(x_train, y_train)
    y_pred_before = baseline_model.predict(x_train)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_pred_before, alpha=0.5, label='Original Data')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='red', label='Ideal Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Before Outlier Removal: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots_IOR/before_outlier_removal.png")
    plt.close()

    # Outlier removal with visualization
    final_x, final_y = iterative_regression_outliers(x_train, y_train, visualize=True)

    # Final model
    model = linear_model.Lasso(alpha=3.5, fit_intercept=True)
    model.fit(final_x, final_y)
    y_pred = model.predict(final_x)

    print("Final model coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Cross-validation scores
    scores = cross_val_score(model, final_x, final_y, cv=5, scoring='r2')
    print("Cross-Validation R2 Scores:", scores)

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, final_x, final_y, cv=5)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training Size')
    plt.ylabel('Score (R²)')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots_IOR/learning_curve.png")
    plt.close()

    # Final scatter plot: predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(final_y, y_pred, alpha=0.5)
    plt.plot([min(final_y), max(final_y)], [min(final_y), max(final_y)], '--', color='red')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("After Outlier Removal: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("plots_IOR/after_outlier_removal.png")
    plt.close()

    # Feature importance
    plt.figure()
    plt.barh(feature_names, model.coef_)
    plt.xlabel("Coefficient Value")
    plt.title("Feature Importance (Lasso Coefficients)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("plots_IOR/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    main()
