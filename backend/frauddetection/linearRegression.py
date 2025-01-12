from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from category_encoders import TargetEncoder

def train_linear_regression(train_data, test_data):
   train_data = train_data.fillna(0) 
   test_data = test_data.fillna(0)

   train_data['day_of_week'] = (train_data['day'] - 1) % 7  # Monday=0, Sunday=6
   train_data['is_weekend'] = (train_data['day_of_week'] >= 5).astype(int)
   train_data['is_business_hours'] = train_data['hour'].between(9, 17).astype(int)

   test_data['day_of_week'] = (test_data['day'] - 1) % 7  # Monday=0, Sunday=6
   test_data['is_weekend'] = (test_data['day_of_week'] >= 5).astype(int)
   test_data['is_business_hours'] = test_data['hour'].between(9, 17).astype(int)

   columns_to_drop = ['minute', 'second']
   train_data = train_data.drop(columns=columns_to_drop, axis=1)
   test_data = test_data.drop(columns=columns_to_drop, axis=1)

   X = train_data.drop('fraud_label', axis=1)
   y = train_data['fraud_label']

   test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
   X_test = test_data.drop('fraud_label', axis=1)
   y_test = test_data['fraud_label']

   categorical_features = ['uyruk', 'hesap_acilis_tipi', 'meslek', 'ikamet_ili']

   encoder = TargetEncoder(cols=categorical_features)
   X[categorical_features] = encoder.fit_transform(X[categorical_features], y)
   X_test[categorical_features] = encoder.transform(X_test[categorical_features])

   adasyn = ADASYN(random_state=42)
   X_train_res, y_train_res = adasyn.fit_resample(X, y)
   model = LinearRegression()

   model.fit(X_train_res, y_train_res)
   y_pred = model.predict(X_test)
   y_pred_class = [1 if prob >= 0.85 else 0 for prob in y_pred]

   print("Linear Regression Report:")
   print(classification_report(y_test, y_pred_class))
   print("Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred_class))
