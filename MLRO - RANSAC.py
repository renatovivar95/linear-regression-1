#!/usr/bin/env python3

"""
MACHINE LEARNING - RENATO VIVAR
RANSAC REGRESSION FOR TOXIC ALGAE PREDICTION
This code implements a RANSAC regression model to predict toxic algae levels based on environmental factors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RANSACRegressor

def main():
    # Create folder for saving plots
    output_dir = "plots_RANSAC"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    x_train = np.load('DATA/X_train.npy')
    y_train = np.load('DATA/y_train.npy')

    feature_names = ['AIR TEMPERATURE', 'WATER TEMP', 'WIND SPEED', 'WIND DIRECTION', 'ILUMINATION']
    dataframeX = pd.DataFrame(x_train, columns=feature_names)
    dataframeY = pd.DataFrame(y_train, columns=['TOXIC ALGUE'])

    # Split (optional)
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Fit RANSAC
    base_estimator = linear_model.Lasso(alpha=3.5, fit_intercept=True)
    ransac = RANSACRegressor(base_estimator, max_trials=100, min_samples=25, residual_threshold=2.25)
    ransac.fit(x_train, y_train)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    Y_pred = ransac.predict(x_train)

    X_inliers = x_train[inlier_mask]
    Y_inliers = y_train[inlier_mask]

    X_outliers = x_train[outlier_mask]
    Y_outliers = y_train[outlier_mask]

    # Retrain on inliers
    model = linear_model.Lasso(alpha=3.5, fit_intercept=True)
    model.fit(X_inliers, Y_inliers)
    Y_final_pred = model.predict(x_train)

    print("Final model coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Plot 1: Inliers, Outliers and RANSAC Prediction
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(y_train)), y_train, color='blue', label='All Data')
    plt.scatter(np.arange(len(y_train))[inlier_mask], y_train[inlier_mask], color='green', label='Inliers')
    plt.scatter(np.arange(len(y_train))[outlier_mask], y_train[outlier_mask], color='red', label='Outliers')
    plt.plot(np.arange(len(y_train)), Y_pred, color='black', label='RANSAC Prediction')
    plt.xlabel("Sample Index")
    plt.ylabel("Toxic Algae Level")
    plt.title("RANSAC Regression - Inliers and Outliers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ransac_inliers_outliers.png")
    plt.close()

    # Plot 2: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, Y_final_pred, alpha=0.5)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='red')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual (Lasso after RANSAC)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predicted_vs_actual.png")
    plt.close()

    # Evaluation metrics
    print(f"Number of inliers: {np.sum(inlier_mask)}")
    print(f"Number of outliers: {np.sum(outlier_mask)}")
    print("Residual sum of squares (MSE): %.4f" % np.mean((Y_final_pred - y_train) ** 2))
    print("R2-score on full training set: %.4f" % r2_score(y_train, Y_final_pred))

    scores = cross_val_score(model, X_inliers, Y_inliers, cv=5, scoring='r2')
    print("Cross-Validation R2 Scores:", scores)
    print("Mean CV R2 Score: %.4f" % np.mean(scores))

if __name__ == "__main__":
    main()
