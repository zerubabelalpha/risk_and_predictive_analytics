# --------------------------------------------------------------
# 📌 Import Required Libraries
# --------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler



# --------------------------------------------------------------
# 📌 2. Encoding Categorical Variables
# --------------------------------------------------------------
def encoder(method, dataframe, columns_label=None, columns_onehot=None):
    """
    Encodes categorical variables using:
    - Label Encoding
    - One-Hot Encoding
    """

    df = dataframe.copy()
    columns_label = columns_label or []
    columns_onehot = columns_onehot or []

    # ---------------------------
    # LABEL ENCODER
    # ---------------------------
    if method == "labelEncoder":
        for col in columns_label:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    # ---------------------------
    # ONE-HOT ENCODER
    # ---------------------------
    elif method == "oneHotEncoder":
        df = pd.get_dummies(
            data=df,
            columns=[c for c in columns_onehot if c in df.columns],
            drop_first=True,
            dtype="int8"
        )
        return df

    return df


# --------------------------------------------------------------
# 📌 3. Scaling Numerical Variables
# --------------------------------------------------------------
def scaler(method, dataframe, columns_scaler):
    """
    Scales numerical features using:
    - StandardScaler
    - MinMaxScaler
    - Log Transformation
    """

    df = dataframe.copy()
    columns_scaler = [c for c in columns_scaler if c in df.columns]

    # ---------------------------
    # STANDARD SCALER
    # ---------------------------
    if method == "standardScaler":
        scaler = StandardScaler()
        df[columns_scaler] = scaler.fit_transform(df[columns_scaler])
        return df

    # ---------------------------
    # MIN-MAX SCALER
    # ---------------------------
    elif method == "minMaxScaler":
        scaler = MinMaxScaler()
        df[columns_scaler] = scaler.fit_transform(df[columns_scaler])
        return df

    # ---------------------------
    # LOG TRANSFORMATION
    # ---------------------------
    elif method == "npLog":
        # Protect against negatives
        for col in columns_scaler:
            df[col] = np.where(df[col] < 0, 0, df[col])
        df[columns_scaler] = np.log1p(df[columns_scaler])
        return df

    return df
