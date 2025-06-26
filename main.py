import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import  warnings
warnings.filterwarnings("ignore")

# Load CSV dataset
df = pd.read_csv("C:/Users/jayan/Downloads/AirPassengers.csv")  # Replace with your actual file name if different

# Parse 'Month' as datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set 'Month' as index
df.set_index('Month', inplace=True)
df = df.asfreq('MS')

# Optional: rename column if needed
df.rename(columns={df.columns[0]: 'Passengers'}, inplace=True)  # Use this only if column is unnamed

# View the data
print(df.head())
print(df.index.min(), "to", df.index.max())


df['Passengers'].plot(title="Monthly Airline Passengers", figsize=(10,5))
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.show()


model = SARIMAX(df['Passengers'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.predict(start=len(df), end=len(df)+11)
print("Forecasted passenger numbers:\n", forecast)

# Plot actual vs forecast
plt.figure(figsize=(10, 5))
df['Passengers'].plot(label='Actual')
forecast.plot(label='Forecast', style='--')
plt.title("Airline Passenger Forecast (Next 12 Months)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.show()

# --- Part 6: Evaluate Model (MAE and RMSE) ---

# Split the dataset: 90% train, 10% test
train_size = int(len(df) * 0.9)
train = df['Passengers'][:train_size]
test = df['Passengers'][train_size:]

# Build and fit model on training data
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Predict for test period
pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))

print("\n📊 Model Evaluation on Test Data:")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")


plt.figure(figsize=(10, 5))
train.plot(label='Train Data')
test.plot(label='Actual Test Data')
pred.plot(label='Predicted Test Data', style='--')

plt.title("SARIMA Forecast: Actual vs Predicted (Test Data)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
