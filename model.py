# model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
dataset = pd.read_csv('CO22317_bike_sales.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Scale data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()

# Train model
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_train_scaled, y_train_scaled)

# Save model and scalers
joblib.dump(regressor, 'bike_model.pkl')
joblib.dump(sc_X, 'scaler_x.pkl')
joblib.dump(sc_y, 'scaler_y.pkl')

# Evaluate
y_pred_scaled = regressor.predict(X_test_scaled)
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_scaled, y_pred_scaled)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
