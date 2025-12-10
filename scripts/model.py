import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

class InsuranceModeler:
    """
    A class for building and evaluating insurance risk and pricing models.
    
    Tasks:
    1. Claim Severity Prediction (Regression): Predict TotalClaims for policies with claims.
    2. Claim Probability Prediction (Classification): Predict HasClaim (0/1).
    3. Premium Optimization: Calculate risk-based premiums.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
    def preprocess_data(self, target_col: str, drop_cols: list = None, cat_cols: list = None, num_cols: list = None):
        """
        Prepares features (X) and target (y) for modeling.
        Handles missing values, encoding, and scaling.
        """
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols, errors='ignore')
            
        y = self.df[target_col]
        X = self.df.drop(columns=[target_col], errors='ignore')
        
        # Identify columns if not provided
        if cat_cols is None:
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if num_cols is None:
            num_cols = X.select_dtypes(include=['number']).columns.tolist()
            
        print(f"Features: {len(X.columns)} total ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for SHAP compatibility
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ])
            
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after encoding
        try:
            cat_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
            self.feature_names = num_cols + list(cat_feature_names)
        except AttributeError:
             self.feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
             
        return pd.DataFrame(X_processed, columns=self.feature_names), y

    def split_data(self, X, y, test_size=0.3, random_state=42):
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_severity_models(self, X_train, y_train):
        """
        Train regression models for Claim Severity (TotalClaims).
        Models: Linear Regression, Random Forest, XGBoost.
        """
        print("\nTraining Severity Models (Regression)...")
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"✓ Trained {name}")
            
        self.models['severity'] = trained_models
        return trained_models

    def evaluate_severity_models(self, X_test, y_test):
        """Evaluate severity models using RMSE and R-squared."""
        results = []
        for name, model in self.models.get('severity', {}).items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({"Model": name, "RMSE": rmse, "R2": r2})
            
        return pd.DataFrame(results)

    def train_claim_probability_models(self, X_train, y_train):
        """
        Train classification models for Claim Probability (HasClaim).
        Models: Logistic Regression, Random Forest, XGBoost.
        """
        print("\nTraining Claim Probability Models (Classification)...")
        # Ensure y is proper integer/binary type
        y_train = y_train.astype(int)
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"✓ Trained {name}")
            
        self.models['probability'] = trained_models
        return trained_models

    def evaluate_probability_models(self, X_test, y_test):
        """Evaluate classification models."""
        y_test = y_test.astype(int)
        results = []
        for name, model in self.models.get('probability', {}).items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
            
        return pd.DataFrame(results)

    def get_feature_importance(self, model_type='severity', model_name='XGBoost'):
        """
        Get feature importance for tree-based models.
        """
        model = self.models.get(model_type, {}).get(model_name)
        if not model:
            print(f"Model {model_name} not found in {model_type}.")
            return None
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
            return feature_imp
        else:
            print(f"Model {model_name} does not support default feature importance.")
            return None

    def explain_model_shap(self, X_sample, model_type='severity', model_name='XGBoost'):
        """
        Use SHAP values to explain model predictions.
        """
        model = self.models.get(model_type, {}).get(model_name)
        if not model:
            return
            
        print(f"Generating SHAP summary for {model_name}...")
        
        # TreeExplainer is best for XGB/RF
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance ({model_name})")
            plt.show() # Display plot
        except Exception as e:
            print(f"SHAP explanation failed: {e}")

    def calculate_risk_premium(self, X_data, severity_model_name='XGBoost', prob_model_name='XGBoost', expense_loading=0.1, profit_margin=0.05):
        """
        Calculate Risk-Based Premium.
        Premium = (Predicted Prob * Predicted Severity) / (1 - Expense - Profit)
        Note: Simplified formula. Often: PurePremium = Freq * Sev. GrossPrem = PurePrem / (1 - Loadings).
        """
        sev_model = self.models['severity'][severity_model_name]
        prob_model = self.models['probability'][prob_model_name]
        
        # 1. Predict Probability of Claim
        # prob_model.predict_proba returns [prob_0, prob_1]
        probs = prob_model.predict_proba(X_data)[:, 1]
        
        # 2. Predict Severity (Expected Claim Amount IF claim occurs)
        severities = sev_model.predict(X_data)
        
        # 3. Calculate Pure Premium
        pure_premiums = probs * severities
        
        # 4. Add Loadings (Expenses + Profit)
        # Using the formula: Gross = Pure / (1 - (Expense% + Profit%))
        # Or additive: Gross = Pure + FixedExpense + ProfitMargin*Pure
        
        # Let's use the division approach for variable loading
        total_loading = expense_loading + profit_margin
        gross_premiums = pure_premiums / (1 - total_loading)
        
        return pd.DataFrame({
            'PredictedProbability': probs,
            'PredictedSeverity': severities,
            'PurePremium': pure_premiums,
            'CalculatedPremium': gross_premiums
        })
