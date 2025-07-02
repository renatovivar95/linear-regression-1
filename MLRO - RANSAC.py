# -*- coding: utf-8 -*-
"""

LAB MACHINE LEARNING - P1 - RENATO VIVAR

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.model_selection import cross_val_score

#UPLOAD THE GIVEN FILES
x_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('X_test.npy')

#TRANSFORM THE ARRAYS TO A DATAFRAME WITH PANDAS
dataframeX = pd.DataFrame(x_train, columns=['AIR TEMPERATURE','WATER TEMP','WIND SPEED','WIND DIRECTION','ILUMINATION'])
dataframeY = pd.DataFrame(y_train, columns=['TOXIC ALGUE'])



#TRAINING AND TESTING DIVISION
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


#REGRESION WITH TRAIN-TEST DIVISION
regr = linear_model.Lasso(alpha=3.5,  fit_intercept=True)
#regr.fit(train_x, train_y)


# Create the RANSAC regressor
ransac = RANSACRegressor(regr, max_trials=100, min_samples=25, residual_threshold=2.25)

# Fit the RANSAC model
ransac.fit(x_train, y_train)

# Identify the inliers and outliers
inlier_mask = ransac.inlier_mask_  # Boolean array, True for inliers, False for outliers
outlier_mask = np.logical_not(inlier_mask)

Y_pred = ransac.predict(x_train)

# Separate inliers and outliers
X_inliers = x_train[inlier_mask]
Y_inliers = y_train[inlier_mask]

X_outliers = x_train[outlier_mask]
Y_outliers = y_train[outlier_mask]

# You can now retrain a model using only the inliers
model = linear_model.Lasso(alpha=3.5,  fit_intercept=True)
model.fit(X_inliers, Y_inliers)
print( model.coef_)  # sparse matrix
print( model.intercept_ )  # probably also relevant

# Predict values on the original dataset or test data
Y_final_pred = model.predict(x_train)

# Example for plotting
plt.scatter(np.arange(len(y_train)), y_train, color='blue', label='Data')
plt.scatter(np.arange(len(y_train))[inlier_mask], y_train[inlier_mask], color='green', label='Inliers')
plt.scatter(np.arange(len(y_train))[outlier_mask], y_train[outlier_mask], color='red', label='Outliers')
plt.plot(np.arange(len(y_train)), Y_pred, color='black', label='RANSAC Prediction')
plt.legend()
plt.show()


# Print the number of inliers and outliers
print(f"Number of inliers: {np.sum(inlier_mask)}")
print(f"Number of outliers: {np.sum(outlier_mask)}")


#REVIEW OF THE R2 SCORE USING TRAINING AND TESTING DIVISION
#print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_train)))
print("Residual sum of squares (MSE): %.2f" % np.mean((Y_final_pred - y_train) ** 2))
print("R2-score: %.2f" % r2_score(y_train , Y_final_pred) )
scores = cross_val_score(model, X_inliers, Y_inliers, cv=5, scoring='r2')
print("Cross-Validation R2 Scores:", scores)



# Print the number of inliers and outliers
print(f"Number of inliers: {np.sum(inlier_mask)}")
print(f"Number of outliers: {np.sum(outlier_mask)}")



#y_pred = regr.predict(test_x)


#REVIEW OF THE R2 SCORE USING TRAINING AND TESTING DIVISION
#print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_train)))
#print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_train) ** 2))
#print("R2-score: %.2f" % r2_score(test_y , y_pred) )





