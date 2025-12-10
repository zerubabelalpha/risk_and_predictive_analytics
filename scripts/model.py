# Linear Regression is a parametric model that fits a straight line (or plane)
# to predict a continuous target variable.
from sklearn.linear_model import LinearRegression

# Decision Tree Regressor is a non-parametric model that splits data into rules.
from sklearn.tree import DecisionTreeRegressor

# Random Forest is an ensemble model (multiple trees combined)
# to reduce overfitting and improve accuracy.
from sklearn.ensemble import RandomForestRegressor

# XGBoost is a powerful Gradient Boosting algorithm that builds trees sequentially
# and corrects the errors of previous trees.
import xgboost as xgb

# train_test_split divides the dataset into training (learn) and testing (evaluate) sets.
from sklearn.model_selection import train_test_split

# Evaluation metrics for regression:
# MAE = Mean Absolute Error → average absolute difference.
# MSE = Mean Squared Error → penalizes big errors.
# R² Score → explains how much variance is captured by the model.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------------------
# 📌 1. FUNCTION: Split Data
# --------------------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features (X) and target (y) into training and testing sets.
    
    - test_size=0.2 → 20% goes to testing, 80% used for training.
    - random_state ensures reproducibility (same split every run).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# --------------------------------------------------------------
# 📌 2. FUNCTION: Train All Models
# --------------------------------------------------------------
def train_models(X_train, y_train):
    """
    Initializes and trains four different regression models.
    Each model learns patterns from X_train to predict y_train.
    """

    # ---------- Initialize Models ----------
    # Linear Regression → simple baseline model
    lr_model = LinearRegression()

    # Decision Tree → learns hierarchical rules (if/else)
    dt_model = DecisionTreeRegressor(random_state=42)

    # Random Forest → multiple trees averaged together (reduces overfitting)
    rfr_model = RandomForestRegressor(random_state=42)

    # XGBoost → boosting method that iteratively improves predictions
    xgb_model = xgb.XGBRegressor(random_state=42)

    # ---------- Train (Fit) Models ----------
    # Each .fit() step is the "learning" process.
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rfr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Return all trained models for evaluation/comparison
    return lr_model, dt_model, rfr_model, xgb_model


# --------------------------------------------------------------
# 📌 3. FUNCTION: Evaluate a Single Model
# --------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Predicts on test set and computes:
    - MAE  → average absolute error
    - MSE  → average squared error
    - R²   → goodness of fit (1 = perfect)
    """

    # Make predictions using the trained model
    y_pred = model.predict(X_test)

    # Calculate regression evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)   # Lower = better
    mse = mean_squared_error(y_test, y_pred)    # Lower = better
    r2 = r2_score(y_test, y_pred)               # Higher = better (max = 1)

    return mae, mse, r2, y_pred


# --------------------------------------------------------------
# 📌 4. FUNCTION: Plot Model Comparison
# --------------------------------------------------------------
def plot_metrics(models, mae_scores, mse_scores, r2_scores):
    """
    Visualizes model performance using bar charts.
    Helps compare:
    - MAE accuracy
    - MSE error
    - R² fit score
    """

    import matplotlib.pyplot as plt

    # ---------- Plot MAE ----------
    plt.figure(figsize=(6, 4))
    plt.bar(models, mae_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of MAE Scores')
    plt.xticks(rotation=45)
    plt.show()

    # ---------- Plot MSE ----------
    plt.figure(figsize=(6, 4))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE Scores')
    plt.xticks(rotation=45)
    plt.show()

    # ---------- Plot R² ----------
    plt.figure(figsize=(6, 4))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.title('Comparison of R-squared Scores')
    plt.xticks(rotation=45)
    plt.show()
