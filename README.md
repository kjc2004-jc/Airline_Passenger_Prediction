# ✈️ Airline Passenger Prediction

This project uses historical airline passenger data to forecast future monthly passenger numbers using a SARIMA model.

---

## 🎯 Objective

To develop a machine learning-based forecasting model that accurately predicts the number of airline passengers. This helps in:

- Forecasting demand
- Optimizing resource allocation
- Improving operational efficiency

---

## 📂 Dataset

- **Source**: [Kaggle - AirPassengers dataset](https://www.kaggle.com/datasets/rakannimer/air-passengers)
- **Time Period**: January 1949 – December 1960
- **Columns**:
  - `Month`: Monthly timestamp
  - `Passengers`: Number of passengers in that month

---

## 🧠 ML Model

We used the **SARIMA (Seasonal ARIMA)** model with the following parameters:

- `order = (1,1,1)`
- `seasonal_order = (1,1,1,12)`

SARIMA is well-suited for time series data with seasonal patterns.

---

## 📊 Results

- **MAE** (Mean Absolute Error): *12.14*
- **RMSE** (Root Mean Squared Error): *15.45*

> These values indicate a reasonably accurate forecast model.

---

## 📈 Visualizations

| Plot | Description |
|------|-------------|
| ![Forecast Plot](screenshots/forecast.png) | Forecasted vs. actual passengers |
| ![Test Prediction](screenshots/test_prediction.png) | Model performance on test data |

---

## 💻 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/Airline_Passenger_Prediction.git
