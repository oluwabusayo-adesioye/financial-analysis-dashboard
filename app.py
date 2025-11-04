import sys
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import altair as alt

#Dashboard configuration
st.set_page_config(page_title="Live Market Research Dashboard", layout="wide")
st.title("Live Market Research Dashboard")

#User input section
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter a stock ticker (eg AAPL, MSFT, TSLA):", value="AAPL")

#Add date/interval control
period = st.sidebar.selectbox("Select data period:", ["7d", "1mo", "3mo", "6mo", "1y"], index=2)
interval = st.sidebar.selectbox("Select interval:", ["15m", "1h", "1d"], index=1)

first_ticker = ticker.split()[0]

#data download
@st.cache_data
def load_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, group_by="column", threads=False)

data = load_data(first_ticker, period, interval)

# Flatten multi-index columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

#Create dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Volatility", "Technical Indicators", "Benchmark"])

with tab1:
    #raw data
    st.subheader("Market Overview")
    st.dataframe(data.tail(25))
    
    #calculate returns, annualized volatility and moving avaerages
    data['Returns'] = data['Close'].pct_change()
    data['Rolling Volatility'] = data['Returns'].rolling(window=50).std() * (1638**0.5)
    data['MA_50'] = data['Close'].rolling(window=325).mean()
    data['MA_200'] = data['Close'].rolling(window=1300).mean()
    
    #display metrics
    st.subheader("Key Market Indicators")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", f"${float(data['Close'].values[-1]):.2f}")
    col2.metric("Rolling Volatility (annualized)", f"{float(data['Rolling Volatility'].values[-1]):.2%}")
    col3.metric("Last Return", f"{float(data['Returns'].values[-1]):.2%}")
    
    #Visualisation
    st.subheader("Price Trend with Moving Averages")
    
    fig = px.line(data, y=['Close', 'MA_50', 'MA_200'], title=f"{ticker} Price Trend")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    #rolling volatility visualisation
    st.subheader("Rolling Volatility Trend")
    st.line_chart(data[['Rolling Volatility']]) #encased in double square brackets so it brings out a df not a series

with tab3:    
    #RSI
    st.subheader("Relative Strength Index (RSI)")
    window_length = 14
    price_diff = data['Close'].diff() #price change/differences
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)
    avg_gain = gain.ewm(com=window_length - 1, min_periods=window_length).mean() #exponential weighted moving average
    avg_loss = loss.ewm(com=window_length - 1, min_periods=window_length).mean() 
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    
    #plot RSI
    st.line_chart(data[["RSI"]])
    st.caption("RSI > 70 = Overbought | RSI < 30 = Oversold")
    
    #bollinger bands
    st.subheader("Bollinger Bands")
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['MA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['MA_20'] - (data['Close'].rolling(window=20).std() * 2)
    
    #plot BB with altair
    date_col_name = data.index.name
    plot_data = data[['Close', 'MA_20', 'Upper_Band', 'Lower_Band']].reset_index() # First, get the date from the index into a column
    # Now, "melt" it
    plot_data_long = plot_data.melt(
        id_vars=[date_col_name],  #X-axis
        value_vars=['Close', 'MA_20', 'Upper_Band', 'Lower_Band'],
        var_name='Indicator',  #Legend
        value_name='Price'      #Y-axis
    )
    
    chart = alt.Chart(plot_data_long).mark_line().encode(
        x=alt.X(f'{date_col_name}:T', title='Date'),
        y=alt.Y('Price:Q', title='Price', scale=alt.Scale(zero=False)),
        color=alt.Color('Indicator:N', title='Legend'),
        tooltip=[date_col_name, 'Indicator', 'Price']).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    #MACD
    st.subheader("MACD(Moving Average Convergence Divergence)")
    short_runner = data['Close'].ewm(span=12, adjust=False).mean()
    long_runner = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_runner - long_runner
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal_Line']
    
    #plot MACD
    date_col_name_macd = data.index.name
    plot_data_macd = data.reset_index()
    base = alt.Chart(plot_data_macd).encode(x=alt.X(f'{date_col_name_macd}:T', title='Date'))
    bar_chart = base.mark_bar().encode(y=alt.Y('Histogram:Q', title='Histogram'),
                                       color=alt.condition(alt.datum.Histogram > 0,
                                                           alt.value("green"),
                                                           alt.value("red")
                                                          )
                                      ).properties(title="MACD Histogram")
    line_data = plot_data_macd.melt(id_vars=[date_col_name_macd],
                                    value_vars=['MACD', 'Signal_Line'],
                                    var_name='Indicator',
                                    value_name='Value'
                                   )
    line_chart = alt.Chart(line_data).mark_line().encode(x=alt.X(f'{date_col_name_macd}:T'),
                                                         y=alt.Y('Value:Q', title='MACD'),
                                                         color=alt.Color('Indicator:N', title='Legend')
                                                        ).properties(title="MACD and Signal Line")
    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(bar_chart, use_container_width=True)
    
    #quick technical summary
    st.subheader("Quick Technical Summary")
    latest_rsi = data['RSI'].iloc[-1]
    latest_vol = data['Rolling Volatility'].iloc[-1]
    latest_macd = data['MACD'].iloc[-1] - data['Signal_Line'].iloc[-1]
    
    st.write(f"**RSI:** {latest_rsi:.2f}")
    st.write(f"**Annualized Volatility:** {latest_vol:.2%}")
    st.write(f"**MACD Signal:** {'Bullish' if latest_macd > 0 else 'Bearish'}")

with tab4:
    st.subheader("Stock Return Comparison with S&P 500")
    #to compare any chosen stock to s&p 500
    benchmark_symbol = "^GSPC"
    benchmark_data = load_data(benchmark_symbol, period, interval)
    
    #flattening again
    if isinstance(benchmark_data.columns, pd.MultiIndex):
        benchmark_data.columns = [col[0] for col in benchmark_data.columns]
    
    #returns for both chosen stock and s&p
    data["Return"] = data["Close"].pct_change()
    benchmark_data["Return"] = benchmark_data["Close"].pct_change()
    
    #put both return columns into one df for comparison and plotting
    comparison = pd.DataFrame({f"{first_ticker} Return": data["Return"], "S&P 500 Return": benchmark_data["Return"]}).dropna()
    
    st.line_chart(comparison)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Oluwabusayo Adesioye**")
