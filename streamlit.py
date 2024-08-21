import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt

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