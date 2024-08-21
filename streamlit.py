import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

@st.cache_data
def get_sp500_data():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    # Create a dictionary mapping company names to symbols
    company_symbol_dict = dict(zip(sp500_table['Security'], sp500_table['Symbol']))
    return company_symbol_dict

st.title('Stock Predictor')
company_symbol_dict = get_sp500_data()
company_name = st.selectbox('Select the company name',list(company_symbol_dict.keys()))
selected_symbol = company_symbol_dict[company_name]

data = yf.download(tickers=selected_symbol, period='5y', interval='1d')

st.write(f"Showing data for {company_name} ({selected_symbol})")
st.write(data.tail())
def remove_outliers(df):
    z_scores = stats.zscore(df.select_dtypes(include=[np.number]))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[filtered_entries]

st.title("Stock Data Visualization")
columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
data = data[columns_to_keep]

# Remove outliers from the data
data_cleaned = remove_outliers(data)

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
plt.show()
st.pyplot(plt)

data_cleaned['Daily Return']=data_cleaned['Close'].pct_change()
plt.figure(figsize=(12, 9))
plt.hist(data_cleaned['Daily Return'].dropna(), bins=50)
plt.xlabel('Daily Return')
plt.ylabel('Counts')
plt.title(f'{company_name} Daily Return Histogram')
st.pyplot(plt)

selected_year = st.selectbox("Select the year:", list(range(2018, datetime.now().year + 1)))

# Fetch data for the selected year
start_date = datetime(selected_year, 1, 1)
end_date = datetime(selected_year + 1, 1, 1)
data = yf.download(tickers=selected_symbol, start=start_date, end=end_date)

# Calculate daily returns and standard deviations
daily_returns = data['Close'].pct_change().dropna()
std_dev_closing = data['Close'].std()
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