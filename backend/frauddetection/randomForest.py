import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np
from imblearn.over_sampling import ADASYN
from category_encoders import TargetEncoder
import pickle
from frauddetection.prepareData import get_labeled_data

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.threshold = 0.5 

    def preprocess(self, data):
        data = data.fillna(0)
        data['transfer_time'] = pd.to_datetime(data['transfer_time'])
        data['hesap_acilis_tarihi'] = pd.to_datetime(data['hesap_acilis_tarihi'])

        if data['transfer_time'].dt.tz is None:
            data['transfer_time'] = data['transfer_time'].dt.tz_localize('UTC') 
        else:
            data['transfer_time'] = data['transfer_time'].dt.tz_convert('UTC') 


        if data['hesap_acilis_tarihi'].dt.tz is None:
            data['hesap_acilis_tarihi'] = data['hesap_acilis_tarihi'].dt.tz_localize('UTC')  
        else:
            data['hesap_acilis_tarihi'] = data['hesap_acilis_tarihi'].dt.tz_convert('UTC')  

        data['year'] = data['transfer_time'].dt.year
        data['month'] = data['transfer_time'].dt.month
        data['day'] = data['transfer_time'].dt.day
        data['hour'] = data['transfer_time'].dt.hour
        data['minute'] = data['transfer_time'].dt.minute
        data['second'] = data['transfer_time'].dt.second
        data['day_of_week'] = (data['day'] - 1) % 7  # Monday=0, Sunday=6
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_business_hours'] = data['hour'].between(9, 17).astype(int)
        

        data['time_diff'] = data['transfer_time'] - data['hesap_acilis_tarihi']
        data['time_diff'] = data['time_diff'].dt.total_seconds()

        columns_to_drop = ['transfer_time', 'hesap_acilis_tarihi', 'year', 'month', 'second', 'minute']
        data = data.drop(columns=columns_to_drop, axis=1)

        return data

    def train(self, train_data, test_data):
        train_data = self.preprocess(train_data)
        test_data = self.preprocess(test_data)

        X_train = train_data.drop('fraud_label', axis=1)
        y_train = train_data['fraud_label']

        X_test = test_data.drop('fraud_label', axis=1)
        y_test = test_data['fraud_label']

        numerical_features = [
            'amount', 'withdrawable_cash', 'bist_tl_cinsinden_hacim',
            'us_borsasi_usd_cinsinden_hacim', 'usd_toplam_islem_hacmi',
            'farkli_kisi_deposit_amount_try', 'farkli_kisi_sayisi', 'hour', 'day'
        ]
        categorical_features = ['uyruk', 'hesap_acilis_tipi', 'meslek', 'ikamet_ili']

        self.encoder = TargetEncoder(cols=categorical_features)
        X_train[categorical_features] = self.encoder.fit_transform(X_train[categorical_features], y_train)
        X_test[categorical_features] = self.encoder.transform(X_test[categorical_features])

        adasyn = ADASYN(random_state=42)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=None, max_features='log2',
            class_weight='balanced', random_state=42
        )
        self.model.fit(X_train_res, y_train_res)

        y_proba = self.model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0.0

        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            current_f1 = f1_score(y_test, y_pred_t)
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.threshold = t


        print(f"Best Threshold: {self.threshold}, F1 Score: {best_f1}")
        y_pred_final = (y_proba >= self.threshold).astype(int)
        print(classification_report(y_test, y_pred_final))


    def predict(self, new_transaction):
        if self.model is None or self.encoder is None:
            raise ValueError("Model and encoder must be trained before prediction.")

        new_transaction = self.preprocess(pd.DataFrame([new_transaction]))
        categorical_features = ['uyruk', 'hesap_acilis_tipi', 'meslek', 'ikamet_ili']
        new_transaction[categorical_features] = self.encoder.transform(new_transaction[categorical_features])

        print("Processed Transaction Data:")
        print(new_transaction)

        probabilities = self.model.predict_proba(new_transaction)
        print("Predicted Probabilities:", probabilities)

        probability = probabilities[:, 1][0]
        print(probability)
        return int(probability >= self.threshold), probability

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(model_path):
        try:
            print(f"Attempting to load model from: {model_path}")
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Training new model...")
            model = FraudDetectionModel()
            
            train_data, test_data = get_labeled_data()
            model.train(train_data, test_data)

            model.save_model(model_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")