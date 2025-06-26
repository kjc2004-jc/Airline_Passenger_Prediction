import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Title
st.title("✈️ Airline Passenger Forecasting")

# Upload CSV
uploaded_file = st.file_uploader("AirPassengers.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df.columns = ['Month', 'Passengers']
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # Plot historical data
    st.subheader("📈 Historical Passenger Data")
    st.line_chart(df['Passengers'])

    # Train-test split
    train_size = int(len(df) * 0.9)
    train = df['Passengers'][:train_size]
    test = df['Passengers'][train_size:]

    # Model training
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecast future
    forecast_period = st.slider("📅 Forecast Months", 1, 24, 12)
    forecast = model_fit.predict(start=len(df), end=len(df) + forecast_period - 1)

    st.subheader(f"🔮 Forecast for Next {forecast_period} Months")
    st.line_chart(forecast)

    # Evaluation
    pred = model_fit.predict(start=test.index[0], end=test.index[-1])
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))

    st.subheader("✅ Model Evaluation")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    # Plot actual vs predicted
    st.subheader("🟦 Actual vs Predicted on Test Data")
    fig, ax = plt.subplots()
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Actual")
    pred.plot(ax=ax, label="Predicted", style='--')
    ax.legend()
    st.pyplot(fig)
