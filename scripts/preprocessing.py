# --------------------------------------------------------------
# 📌 Import Required Libraries
# --------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# --------------------------------------------------------------
# 📌 1. Load and Clean Data
# --------------------------------------------------------------
def load_and_clean_data(filepath):
    """
    Loads raw insurance data and performs:
    - Missing value handling
    - Date feature engineering
    - Boolean conversion
    - Column removal (>80% missing)
    - Duplicate removal
    """

    data = pd.read_csv(filepath)

    # ----------------------------------------------------------
    # Handle missing-like strings
    # ----------------------------------------------------------
    data.replace(['', 'Not specified'], np.nan, inplace=True)

    # ----------------------------------------------------------
    # Date handling
    # ----------------------------------------------------------
    if 'TransactionMonth' in data.columns:
        data['TransactionMonth'] = pd.to_datetime(
            data['TransactionMonth'], errors='coerce'
        )
        data['TransactionYear'] = data['TransactionMonth'].dt.year
        data['TransactionMonthNum'] = data['TransactionMonth'].dt.month
        data.drop('TransactionMonth', axis=1, inplace=True)

    if 'VehicleIntroDate' in data.columns:
        data['VehicleIntroDate'] = pd.to_datetime(
            data['VehicleIntroDate'], errors='coerce'
        )
        data.drop('VehicleIntroDate', axis=1, inplace=True)

    # ----------------------------------------------------------
    # Vehicle age feature
    # ----------------------------------------------------------
    if 'TransactionYear' in data.columns and 'RegistrationYear' in data.columns:
        data['VehicleAge'] = data['TransactionYear'] - data['RegistrationYear']

    # ----------------------------------------------------------
    # Boolean-like columns → binary
    # ----------------------------------------------------------
    bool_cols = [
        'IsVATRegistered', 'AlarmImmobiliser', 'TrackingDevice',
        'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder'
    ]

    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].map(
                {'Yes': 1, 'No': 0, True: 1, False: 0}
            )

    # ----------------------------------------------------------
    # Drop columns with >80% missing values
    # ----------------------------------------------------------
    thresh = len(data) * 0.2
    data = data.dropna(axis=1, thresh=thresh)

    # ----------------------------------------------------------
    # Fill missing categorical values
    # ----------------------------------------------------------
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = data[col].fillna('Unknown')

    # ----------------------------------------------------------
    # Fill missing numeric values
    # ----------------------------------------------------------
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())

    # ----------------------------------------------------------
    # Remove duplicates
    # ----------------------------------------------------------
    data = data.drop_duplicates(keep='first')

    return data


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
