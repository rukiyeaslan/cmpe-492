import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from imblearn.over_sampling import ADASYN


def train_xgboost(train_data, test_data):

    train_data = train_data.fillna(0) 
    test_data = test_data.fillna(0) 

    train_data['day_of_week'] = (train_data['day'] - 1) % 7  # Monday=0, Sunday=6
    train_data['is_weekend'] = (train_data['day_of_week'] >= 5).astype(int)
    train_data['is_business_hours'] = train_data['hour'].between(9, 17).astype(int)

    test_data['day_of_week'] = (test_data['day'] - 1) % 7  # Monday=0, Sunday=6
    test_data['is_weekend'] = (test_data['day_of_week'] >= 5).astype(int)
    test_data['is_business_hours'] = test_data['hour'].between(9, 17).astype(int)

    # Separate features and target
    X = train_data.drop('fraud_label', axis=1)
    y = train_data['fraud_label']


    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    X_test = test_data.drop('fraud_label', axis=1)
    y_test = test_data['fraud_label']

    categorical_features = ['uyruk', 'hesap_acilis_tipi', 'meslek', 'ikamet_ili']

    # encode categorical features
    encoder = TargetEncoder(cols=categorical_features)
    X[categorical_features] = encoder.fit_transform(X[categorical_features], y)

    X = train_data.drop('fraud_label', axis=1)
    y = train_data['fraud_label']

    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    X_test = test_data.drop('fraud_label', axis=1)
    y_test = test_data['fraud_label']

    encoder = TargetEncoder(cols=categorical_features)
    X[categorical_features] = encoder.fit_transform(X[categorical_features], y)
    X_test[categorical_features] = encoder.transform(X_test[categorical_features])

    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X, y)

    # 5. Train the XGBoost Model
    model = XGBClassifier(
        scale_pos_weight=1, 
        use_label_encoder=False, 
        eval_metric="logloss"
    )
    model.fit(X_train_res, y_train_res)

    # 6. Evaluate the Model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")

    return model
