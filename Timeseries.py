import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
data = pd.read_csv('razavi_bread_sales.csv')

# Perform EDA
print(data.head())
print(data.describe())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of sales volume
sns.histplot(data['Sales'])
plt.title('Distribution of Sales Volume')
plt.show()

# Visualize the sales volume over time
plt.plot(data['Expiration Date'], data['Sales'])
plt.title('Sales Volume Over Time')
plt.xlabel('Expiration Date')
plt.ylabel('Sales Volume')
plt.show()

# Prepare the data for modeling
data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
data = data.set_index('Expiration Date')
X = data[['Sales', 'Customers', 'Average Rating', 'Promo Applied', 'New Product', 'Best Seller', 'On Sale']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an ARIMA model
model = ARIMA(y_train, order=(1, 1, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Evaluate the performance of the ARIMA model
y_pred_arima = model_fit.forecast(len(y_test))[0]
mse_arima = mean_squared_error(y_test, y_pred_arima)
mae_arima = mean_absolute_error(y_test, y_pred_arima)
rmse_arima = np.sqrt(mse_arima)
print('ARIMA MSE: ', mse_arima)
print('ARIMA MAE: ', mae_arima)
print('ARIMA RMSE: ', rmse_arima)

# Train a Prophet model
prophet_data = X_train.reset_index().rename(columns={'Expiration Date': 'ds', 'Sales': 'y'})
model_prophet = Prophet()
model_prophet.fit(prophet_data)
future = model_prophet.make_future_dataframe(periods=len(y_test), freq='D')
forecast = model_prophet.predict(future)
y_pred_prophet = forecast[['ds', 'yhat']].tail(len(y_test)).set_index('ds')['yhat'].values
mse_prophet = mean_squared_error(y_test, y_pred_prophet)
mae_prophet = mean_absolute_error(y_test, y_pred_prophet)
rmse_prophet = np.sqrt(mse_prophet)
print('Prophet MSE: ', mse_prophet)
print('Prophet MAE: ', mae_prophet)
print('Prophet RMSE: ', rmse_prophet)

# Train an XGBoost model
model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print('XGBoost MSE: ', mse_xgb)
print('XGBoost MAE: ', mae_xgb)
print('XGBoost RMSE: ', rmse_xgb)

# Select the best model based on RMSE
if rmse_arima <= rmse_prophet and rmse_arima <= rmse_xgb:
    print('ARIMA is the best-performing model.')
    y_pred = y_pred_arima
elif rmse_prophet <= rmse_arima and rmse_prophet <= rmse_xgb:
    print('Prophet is the best-performing model.')
    y_pred = y_pred_prophet
else:
    print('XGBoost is the best-performing model.')
    y_pred = y_pred_xgb

# Visualize the actual vs. predicted sales volume
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title('Actual vs. Predicted Sales Volume')
plt.xlabel('Expiration Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()
