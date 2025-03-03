import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
from typing import Tuple, Optional
import requests

# Page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Alpha Vantage configuration
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "YOUR_KEY")  # Replace with your API key or use Streamlit secrets

@st.cache_data(ttl=3600)
def search_company(query: str) -> list:
    """Search for a company using Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "bestMatches" in data:
            return data["bestMatches"]
        else:
            st.error("No matches found or API limit reached")
            return []
    except Exception as e:
        st.error(f"Error searching for company: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def get_stock_data(symbol: str, exchange: str = "", start_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch historical stock data from Alpha Vantage."""
    # Combine exchange and symbol if exchange is provided
    ticker = f"{exchange}:{symbol}" if exchange and exchange != "None" else symbol
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        
        if "Time Series (Daily)" not in data:
            st.error("Data not available or API limit reached. Check your symbol or try again later.")
            return None
            
        # Parse the data
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        
        # Convert data types
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        # Set index as datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date (ascending)
        df.sort_index(inplace=True)
        
        # Filter by start date if provided
        if start_date:
            df = df[df.index >= start_date]
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

@st.cache_resource
def create_model() -> Sequential:
    """Create a simple LSTM model."""
    model = Sequential([
        LSTM(50, input_shape=(30, 5), return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@st.cache_data
def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """Prepare data for model training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    return scaled_data, scaler

def create_sequences(data: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback, 3])
    return np.array(X), np.array(y)

def predict_prices(model, last_sequence: np.ndarray, scaler: MinMaxScaler, days: int) -> np.ndarray:
    """Predict future prices."""
    predictions = []
    curr_seq = last_sequence.copy()
    
    for _ in range(days):
        curr_pred = model.predict(curr_seq.reshape(1, 30, 5), verbose=0)
        predictions.append(curr_pred[0, 0])
        
        new_row = curr_seq[-1].copy()
        new_row[3] = curr_pred[0, 0]
        curr_seq = np.vstack([curr_seq[1:], [new_row]])
    
    pred_sequence = np.zeros((len(predictions), 5))
    pred_sequence[:, 3] = predictions
    return scaler.inverse_transform(pred_sequence)[:, 3]

def create_forecast_plot(df: pd.DataFrame, predictions: np.ndarray, company_name: str) -> go.Figure:
    """Create visualization of forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'].values[-30:],
                             name='Historical', line=dict(color='#00FF7F', width=2)))
    
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(predictions), freq='B')
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast',
                             line=dict(color='#FF4500', width=2, dash='dash')))
    
    fig.update_layout(title=f"{company_name} - Stock Price Forecast", template="plotly_dark",
                      xaxis_title="Date", yaxis_title="Price")
    return fig

def main():
    st.title("📈 Stock Price Predictor")
    
    # API Key configuration
    with st.expander("⚙️ API Configuration"):
        api_key_input = st.text_input(
            "Alpha Vantage API Key (Leave empty to use the default key)", 
            value="", 
            type="password",
            help="Get your free API key at https://www.alphavantage.co/support/#api-key"
        )
        
        if api_key_input:
            global ALPHA_VANTAGE_API_KEY
            ALPHA_VANTAGE_API_KEY = api_key_input
    
    # Information about the app
    st.markdown("""
    ### 📋 How to use this app
    1. Enter a company name or ticker symbol in the search box
    2. Select the correct company from the search results
    3. Choose your preferred exchange (NSE/BSE for Indian stocks)
    4. Adjust settings as needed
    5. Click "Predict Stock Prices" to generate forecasts
    
    > ⚠️ **Note**: The free Alpha Vantage API has limitations (5 API calls per minute, 500 per day).
    If you encounter errors, please wait and try again.
    """)
    
    # Company search
    company_query = st.text_input("🔍 Search for a company by name or ticker symbol:")
    
    ticker = None
    company_name = None
    exchange = None
    
    if company_query:
        search_results = search_company(company_query)
        
        if search_results:
            # Create options for the selectbox
            options = [f"{r['2. name']} ({r['1. symbol']}) - {r['4. region']}" for r in search_results]
            selected_company = st.selectbox("Select a company from the results:", options)
            
            # Get the index of the selected option
            selected_idx = options.index(selected_company)
            selected_data = search_results[selected_idx]
            
            company_name = selected_data['2. name']
            ticker = selected_data['1. symbol']
            
            # For Indian stocks, choose exchange
            if selected_data['4. region'] == "India":
                exchange = st.selectbox(
                    "Select exchange:",
                    ["None", "NSE", "BSE"],
                    help="For Indian stocks, select NSE (National Stock Exchange) or BSE (Bombay Stock Exchange)"
                )
                st.info("For Indian stocks, you may need to specify the exchange (NSE or BSE)")
            
            # Display company info
            st.markdown(f"""
            **Selected Company**: {company_name}  
            **Symbol**: {ticker}  
            **Region**: {selected_data['4. region']}  
            **Currency**: {selected_data['8. currency']}
            """)
    
    # Settings
    with st.expander("⚙️ Advanced Settings"):
        lookback_days = st.slider("Historical data to use (days)", 365, 1095, 730, 
                                  help="Number of days of historical data to train the model")
        training_epochs = st.slider("Training epochs", 1, 20, 5, 
                                    help="More epochs can improve accuracy but take longer to train")
    
    start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    if ticker and st.button("🔮 Predict Stock Prices"):
        with st.spinner("Fetching historical data..."):
            df = get_stock_data(ticker, exchange, start_date)
            
        if df is not None and not df.empty:
            # Display basic stats
            st.write(f"### 📊 Historical Data for {company_name}")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            pct_change = (price_change / df['Close'].iloc[-2]) * 100
            
            with metrics_col1:
                st.metric("Current Price", f"{current_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
            with metrics_col2:
                st.metric("52-Week High", f"{df['High'].max():.2f}")
            with metrics_col3:
                st.metric("52-Week Low", f"{df['Low'].min():.2f}")
            with metrics_col4:
                st.metric("Avg. Volume", f"{df['Volume'].mean():.0f}")
            
            # Plot historical data
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index[-90:],
                open=df['Open'][-90:],
                high=df['High'][-90:],
                low=df['Low'][-90:],
                close=df['Close'][-90:],
                name="Price"
            ))
            fig.update_layout(title=f"{company_name} - Recent Stock Price History", 
                             template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            
            scaled_data, scaler = prepare_data(df)
            X, y = create_sequences(scaled_data)
            
            if len(X) < 30:
                st.error("Not enough data for prediction. Need at least 31 days of historical data.")
                return
            
            with st.spinner("Training model..."):
                model = create_model()
                model.fit(X, y, epochs=training_epochs, batch_size=32, verbose=0)
            
            timeline = st.selectbox("Select Forecast Timeline", 
                                   ["Next Week (5 days)", "Next 2 Weeks (10 days)", "Next Month (21 days)", 
                                    "Next 2 Months (42 days)", "Next Quarter (63 days)"])
            
            days_map = {
                "Next Week (5 days)": 5, 
                "Next 2 Weeks (10 days)": 10,
                "Next Month (21 days)": 21, 
                "Next 2 Months (42 days)": 42,
                "Next Quarter (63 days)": 63
            }
            
            with st.spinner("Generating forecast..."):
                last_sequence = scaled_data[-30:]
                predictions = predict_prices(model, last_sequence, scaler, days_map[timeline])
                fig = create_forecast_plot(df, predictions, company_name)
                st.plotly_chart(fig, use_container_width=True)
                
                pred_df = pd.DataFrame({
                    'Date': pd.date_range(start=df.index[-1] + timedelta(days=1), 
                                          periods=len(predictions), freq='B'),
                    'Predicted Price': predictions
                })
                st.write("### 📋 Detailed Forecast")
                st.dataframe(pred_df.style.format({'Predicted Price': '{:.2f}'}), use_container_width=True)
                
                # Display prediction summary
                last_price = df['Close'].iloc[-1]
                final_pred = predictions[-1]
                change = final_pred - last_price
                pct_change = (change / last_price) * 100
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Current Price", f"{last_price:.2f}")
                with summary_col2:
                    st.metric(f"Predicted ({timeline.split('(')[0].strip()})", f"{final_pred:.2f}")
                with summary_col3:
                    st.metric("Predicted Change", f"{change:.2f} ({pct_change:.2f}%)", 
                             delta_color="normal" if change >= 0 else "inverse")
                
                st.info("💡 These predictions are based on historical trends and should be used for informational purposes only. Do not make investment decisions solely based on these forecasts.")
                st.warning("⚠️ Past performance is not indicative of future results. Always do your own research before investing.")
        else:
            st.error("Unable to fetch data for the selected company. Please try a different company or check the exchange.")

if __name__ == "__main__":
    main()
