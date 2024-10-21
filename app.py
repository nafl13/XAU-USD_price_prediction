from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Load the pre-trained model and scaler (replace with actual paths)
model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to create features from the close prices


def create_features_from_close(df):
    # Lag Features
    for lag in range(1, 6):
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)

    # 5-Day Moving Average (MA_5)
    df['MA_5'] = df['Close'].rolling(window=5).mean()

    # 5-Day Rate of Change (ROC_5)
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / \
        df['Close'].shift(5) * 100

    # Momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

    # Exponential Moving Average (EMA_5)
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

    # 5-Day Standard Deviation (Volatility)
    df['std_5'] = df['Close'].rolling(window=5).std()

    # Bollinger Bands (Upper and Lower Bands)
    df['Upper_Band'] = df['MA_5'] + (2 * df['std_5'])
    df['Lower_Band'] = df['MA_5'] - (2 * df['std_5'])

    # Close difference (Close_diff)
    df['Close_diff'] = df['Close'] - df['Close'].shift(1)

    # Cumulative return for the last 5 days
    df['Cumulative_Return_5'] = df['Close'].pct_change(5).cumsum()

    # Volatility (alias for standard deviation)
    df['Volatility_5'] = df['std_5']

    return df

# Function to extract date features


def extract_date_features(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return date.year, date.month, date.day, date.weekday()

# Initial load route: Displays "Welcome" when no form is submitted


@app.route('/')
def index():
    # Default to "Welcome" if no prediction
    prediction_text = session.get('prediction_text', "Welcome")
    # Clear the prediction after displaying it
    session.pop('prediction_text', None)
    return render_template('index.html', prediction_text=prediction_text)

# Route for form submission and prediction


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user inputs
    close1 = float(request.form['close1'])
    close2 = float(request.form['close2'])
    close3 = float(request.form['close3'])
    close4 = float(request.form['close4'])
    close5 = float(request.form['close5'])
    close6 = float(request.form['close6'])
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    date_str = request.form['date']

    # Creating DataFrame with the last 6 close prices
    close_prices = [close1, close2, close3, close4, close5, close6]
    df = pd.DataFrame({'Close': close_prices})

    # Create features
    df = create_features_from_close(df)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        # Store a message in the session if data is insufficient
        session['prediction_text'] = "Insufficient data to generate prediction."
        return redirect(url_for('index'))

    # Extract date features (if needed in future models)
    year, month, day, day_of_week = extract_date_features(date_str)

    # Adding high, low, open to the DataFrame
    df['Open'] = open_price
    df['High'] = high_price
    df['Low'] = low_price

    # Selecting columns in correct order
    columns_needed = [
        'Open', 'High', 'Low', 'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4', 'Close_lag5',
        'MA_5', 'ROC_5', 'Momentum_5', 'EMA_5', 'std_5', 'Upper_Band', 'Lower_Band',
        'Volatility_5', 'Cumulative_Return_5', 'Close_diff'
    ]
    df = df[columns_needed]

    # Scaling the features
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Predicting the next close price
    prediction = model.predict(df_scaled)[0]

    # Determine if the market is Bullish or Bearish
    if prediction > close6:
        prediction_text = f"The predicted value is  {prediction:.2f}  - Bullish"
    else:
        prediction_text = f"The predicted value is  {prediction:.2f}  - Bearish"

    # Store the prediction in the session
    session['prediction_text'] = prediction_text

    # Redirect to index
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
