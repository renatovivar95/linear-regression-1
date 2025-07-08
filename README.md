
## 🔧 How to Run

To execute the scripts, use the following commands in your terminal:

```bash
python IOR_lasso_outlier_removal.py
python RANSAC_regression.py
```
## 📁 Output Plots

### `plots_IOR/` (Iterative Outlier Removal with Lasso)

- **outlier_removal_process.png**: Shows the regression fit at different iterations while outliers are removed.
- **after_outlier_removal.png**: Scatter plot comparing predicted vs actual values after cleaning.
- **learning_curve.png**: Shows how model performance changes with training size.
- **feature_importance.png**: Visualizes the Lasso coefficients (feature importance).

### `plots_RANSAC/` (RANSAC + Lasso Regression)

- **ransac_inliers_outliers.png**: Highlights inliers vs outliers identified by RANSAC.
- **predicted_vs_actual.png**: Scatter plot of predicted vs actual values using the final model.
- **ransac_iterations_grid.png**: Visual explanation of how RANSAC iteratively fits models and selects the best one.


The file Machine Learning - LR.pdf contains a summary of the methodology, results, and explanation of both robust regression approaches.


## Theorical Explanation Multiple Linear Regression with Outliers

This is a linear regression problem that illustrates some characteristics of real applications. We have a set of $n=200$ samples with 5 independent variables $x_1, x_2, \ldots, x_p,\ p=5$ and dependent variable $y$:

$$
y = \beta_{0}+\sum_{i=1}^{p} \beta_i X_i
$$

These variables are affected by two main types of noise:
1. Instrument noise $\eta$, which is well approximated by zero-mean white Gaussian noise.
2. Human error $\xi$ that affects about 25% of the samples of the dependent variable, but is negligible in the independent variables.

Therefore, the real data samples are modeled as:

$$
\hat{y}^{(j)} = \beta_{0} + \sum_{i=1}^{p} \beta_i X_i^{(j)} + \eta^{(j)} + \xi^{(j)}
$$

$$
\hat{x_i}^{(j)} = x_i^{(j)} + \eta^{(j)}
$$

---

### Outlier Detection: Boxplot Rule / IQR Method

The first step is detecting and recognizing outliers in the dataset. This preliminary process is called the **Boxplot rule** or **IQR Method**, which calculates the interquartile range (IQR) and marks outliers as points beyond the range:

$$
\text{Lower Bound / Upper Bound} = Q1 \pm 1.5 \times IQR
$$

Where $Q1$ and $Q3$ are the 25th and 75th percentiles of $Y$, respectively [Wilcox, 2023].

---

### Outlier Handling Strategy

Once we identify that the outliers are part of the dataset, different strategies can be applied. The one used here is **capping (or winsorizing)**: replacing outliers in the dependent variable with a fixed upper or lower bound (e.g., 5th or 95th percentile). This preserves the data size while reducing the influence of extreme values.

For the dependent variable $Y$, we implemented a function to remove outliers **one by one**, based on the largest value of quadratic loss $(\hat{y} - y)^2$. In each step, the regression model was re-trained with $(n - 1)$ samples, excluding the sample with the highest loss. After removing 25% of the samples (assumed human errors), we obtained the final model.

---

### Regularization: Lasso Regression (L1)

To improve the model and avoid overfitting, **Lasso Regression** was used. It adds an L1 penalty (sum of absolute values of the regression coefficients) to the loss function of OLS:

$$
\min \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 + \lambda \sum_{i=1}^{p} |\beta_i| \right)
$$

Lasso sets some coefficients to zero, effectively performing variable selection. It's helpful for identifying the most relevant variables and for regularization in noisy datasets.

To select the best value of $\lambda$, **k-fold cross-validation** was used:

1. The dataset is split into $k$ folds.
2. For each $\lambda$, the model is trained on $(k - 1)$ folds and tested on the remaining one.
3. This process is repeated $k$ times.
4. The average error is calculated across all folds.
5. The $\lambda$ minimizing this error is selected [Friedman et al., 2008].

---

### Evaluation Metrics

To evaluate the model performance, **Cross-Validation $R^2$ Score** and **Mean Squared Error (MSE)** were used. Cross-validation helps assess generalization ability and avoid overfitting or underfitting.

Each iteration of cross-validation computes:

- $R^2$ score
- MSE

The final performance is the average across all folds.

---


## Code Overview

This project applies Lasso regression to predict the presence of toxic algae based on environmental features. The script includes data preprocessing, iterative outlier removal, model training, cross-validation, and visualization of results.

## How It Works

1. **Data Loading**  
   The script loads the training datasets (`X_train.npy` and `y_train.npy`) from the `DATA/` directory.

2. **Data Preprocessing**  
   The NumPy arrays are converted into Pandas DataFrames for better handling. The input features include:
   - Air Temperature  
   - Water Temperature  
   - Wind Speed  
   - Wind Direction  
   - Illumination  

3. **Outlier Removal with Iterative Lasso Regression**  
   A custom function performs Lasso regression iteratively. In each iteration:
   - The model is trained on the current dataset.
   - The sample with the highest residual (prediction error) is considered an outlier and removed.
   - This process is repeated 50 times to improve data quality.

4. **Model Training**  
   After cleaning the data, a final Lasso model is trained using `alpha = 3.5`.

5. **Model Evaluation**  
   - The model coefficients and intercept are printed.
   - 5-fold cross-validation is performed to compute R² scores.
   - A **learning curve** plot shows how model performance evolves with increasing training data.
   - A **scatter plot** of predicted vs actual values visualizes prediction accuracy.

## Visualization Outputs
- **Learning Curve**: Shows training and validation scores as the training size increases.
- **Predicted vs Actual Plot**: Helps assess how well the model fits the data by comparing predictions to actual target values.



### References

- Wilcox, R. (2023). *Introduction to Robust Estimation and Hypothesis Testing*.
- Friedman, J., Hastie, T., Tibshirani, R. (2008). *Regularization Paths for Generalized Linear Models via Coordinate Descent*.

