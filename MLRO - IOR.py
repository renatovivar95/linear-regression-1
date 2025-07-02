# -*- coding: utf-8 -*-
"""

LAB MACHINE LEARNING - P1 - RENATO VIVAR

"""
#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import lasso_path


#UPLOAD THE GIVEN FILES
x_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('X_test.npy')

#TRANSFORM THE ARRAYS TO A DATAFRAME WITH PANDAS
dataframeX = pd.DataFrame(x_train, columns=['AIR TEMPERATURE','WATER TEMP','WIND SPEED','WIND DIRECTION','ILUMINATION'])
dataframeY = pd.DataFrame(y_train, columns=['TOXIC ALGUE'])

#FUNCTION TO MAKE ITERATIVE REGRESSION AND OUTLIER REMOVAL
def iterative_regression_outliers(x_data, y_data, alpha=3.5, iterations=50):
    # Initialize training data
    train_x = x_data.copy()
    train_y = y_data.copy()
    
    for i in range(iterations):
        # Create and fit the Lasso model
        model = linear_model.Lasso(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_y)
        
        # Predict values and calculate residuals (squared errors)
        y_pred = model.predict(train_x)
        aux_sse = (y_pred - train_y) ** 2
        
        # Find the index of the largest residual (outlier)
        aux_index = np.argmax(aux_sse)
        
        # Remove the outlier from the training data
        train_x = np.delete(train_x, aux_index, axis=0)
        train_y = np.delete(train_y, aux_index, axis=0)
        
    
    return train_x, train_y

#IMPLEMENTING FUNCTION
final_x, final_y = iterative_regression_outliers(x_train, y_train)

#TRAINING MODEL
model = linear_model.Lasso(alpha=3.5,  fit_intercept=True)
model.fit(final_x, final_y)

y_pred = model.predict(final_x)
print(model.coef_, model.intercept_)

#CROSS VALIDATION 
scores = cross_val_score(model, final_x, final_y, cv=5, scoring='r2')
print("Cross-Validation R2 Scores:", scores)


#PLOTS
train_sizes, train_scores, test_scores = learning_curve(model, final_x, final_y, cv=5)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Score (R² or MSE)')
plt.title('Learning Curve')
plt.legend()
plt.show()

plt.scatter(final_y, y_pred, alpha=0.5)
plt.plot([min(final_y), max(final_y)], [min(final_y), max(final_y)], '--', color='red')  # 45-degree line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()





