import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

@st.cache_data
def get_sp500_data():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    # Create a dictionary mapping company names to symbols
    company_symbol_dict = dict(zip(sp500_table['Security'], sp500_table['Symbol']))
    return company_symbol_dict

@st.cache_data
def download_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, actions=False)

@st.cache_data
def train_lstm_model(train_data, time_step):
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
    return model

@st.cache_data
def get_data_for_year(symbol, year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)
    return yf.download(tickers=symbol, start=start_date, end=end_date)

@st.cache_data
def download_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, actions=False)


st.title('Fortune Teller')
company_symbol_dict = get_sp500_data()
company_name = st.selectbox('Select the company name',list(company_symbol_dict.keys()))
selected_symbol = company_symbol_dict[company_name]



end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)
data = download_stock_data(selected_symbol, start, end)


st.write(f"Showing data for {company_name} ({selected_symbol})")
st.write(data.tail())
def remove_outliers(df):
    z_scores = stats.zscore(df.select_dtypes(include=[np.number]))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[filtered_entries]

st.title("Stock Data Visualization")
columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
data_keep = data[columns_to_keep]

# Remove outliers from the data
data_cleaned = remove_outliers(data_keep)

# Select a feature to visualize
feature = st.selectbox('Select the feature to visualize:', columns_to_keep)
plot_type = st.selectbox("Select plot type:", ["Line Graph", "Scatter Plot"])

# Show the scatter plot for the selected feature
plt.figure(figsize=(12, 6))
if plot_type=="Line Graph":
    plt.plot(data_cleaned.index, data_cleaned[feature], alpha=0.6)
    plt.title(f'{company_name} {feature} Line Graph Without Outliers')
elif plot_type=="Scatter Plot":
    plt.scatter(data_cleaned.index, data_cleaned[feature], alpha=0.6)
    plt.title(f'{company_name} {feature} Scatter Plot With Outliers')
plt.xlabel('Date')
plt.ylabel(feature)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

selected_year = st.selectbox("Select the year:", list(range(2018, datetime.now().year + 1)))

# Fetch data for the selected year
start_date_dev = datetime(selected_year, 1, 1)
end_date_dev = datetime(selected_year + 1, 1, 1)
data_dev = yf.download(tickers=selected_symbol, start=start_date_dev, end=end_date_dev)

# Calculate daily returns and standard deviations
daily_returns = data_dev['Close'].pct_change().dropna()
std_dev_closing = data_dev['Close'].std()
std_dev_rets = daily_returns.std()

# Display the standard deviations
st.write(f"Standard Deviation of Closing Prices ({selected_year}): {std_dev_closing}")
st.write(f"Standard Deviation of Daily Returns ({selected_year}): {std_dev_rets}")

# Plot the daily returns histogram
plt.figure(figsize=(12, 9))
plt.hist(daily_returns, bins=50)
plt.xlabel('Daily Return')
plt.ylabel('Counts')
plt.title(f'Daily Returns for {company_name} in {selected_year}')
st.pyplot(plt)







# Prepare the data for training


df1 = data.reset_index()['Close']
import matplotlib.pyplot as plt
import numpy as np
# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))


# Split dataset into training and testing
training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

# Function to create a dataset matrix
import numpy
@st.cache_data
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# Reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
with st.spinner('Training the LSTM model, please wait...'):
 model = train_lstm_model(train_data, time_step)

X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



import tensorflow as tf
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error

# Calculate RMSE
math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(ytest, test_predict))

# Plotting
st.subheader("Actual vs Predicted Close Prices")
plt.figure(figsize=(12, 6))

# Plot training predictions
look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

# Plot testing predictions
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(df1), color='blue', label='Actual Prices')
plt.plot(trainPredictPlot, color='orange', label='Training Predictions')
plt.plot(testPredictPlot, color='red', label='Testing Predictions')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)



# Future predictions
st.subheader("Future Predictions for the Next 30 Days")
x_input=test_data[340:].reshape(1,0-1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = 100
i = 0
while (i < 30):
    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

# Plot future predictions
day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(df1[-100:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.title("Future Stock Price Predictions for the Next 30 Days")
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.show()
st.pyplot(plt)

df3=df1.tolist()
df3.extend(lst_output)
plt.figure(figsize=(12, 6))
plt.plot(df3[1200:])
plt.title('Recent and Predicted Stock Prices (Scaled)')
plt.xlabel('Time')
plt.ylabel('Scaled Price')
st.pyplot(plt.gcf())

df3=scaler.inverse_transform(df3).tolist()
plt.figure(figsize=(12, 6))
plt.plot(df3)
plt.title('Recent and Predicted Stock Prices (Original Scale)')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(plt.gcf())



# Feature Engineering
data['Daily Return'] = data['Close'].pct_change()
data['Moving Average'] = data['Close'].rolling(window=20).mean()
data.dropna(inplace=True)

# Define features and target
features = ['Daily Return', 'Moving Average']
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1: Buy, 0: Sell

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Split data into training and testing sets
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set and evaluate
y_pred = model.predict(X_test_scaled)

# Preparing to predict for the future
# Use the most recent data to generate features for future prediction
recent_data = data[-20:].copy()  # Make sure to copy to avoid warnings

# Generate features for prediction
recent_data['Daily Return'] = recent_data['Close'].pct_change()
recent_data['Moving Average'] = recent_data['Close'].rolling(window=20).mean()
recent_data.dropna(inplace=True)

# Display recent data and calculated features
st.subheader("Most Recent Data")
st.dataframe(recent_data.tail(10))

# Use only the most recent row for future prediction, keeping it in DataFrame format
latest_features = recent_data[features].iloc[-1:]

# Show the feature values before scaling
st.subheader("Feature Values for Prediction (Unscaled)")
st.dataframe(latest_features)

# Predict using the trained model
with st.spinner('Predicting the future... Please wait...'):
    # Predict using the trained model
    latest_features_scaled = scaler.transform(latest_features)
    future_prediction = model.predict(latest_features_scaled)

# Interpret the prediction
action = "Buy" if future_prediction[0] == 1 else "Sell"

# Show scaled feature values and prediction
st.subheader("Feature Values for Prediction (Scaled)")
st.dataframe(pd.DataFrame(latest_features_scaled, columns=features))

st.markdown(f"## Future prediction for the next day: **{action}**")