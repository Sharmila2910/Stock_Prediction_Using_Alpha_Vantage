
📈 Stock Price Predictor

This is a Stock Price Prediction Web App built with Streamlit, TensorFlow, Pandas, and Plotly.  
It fetches historical stock data using the **Alpha Vantage API**, trains an **LSTM model**, and forecasts future stock prices.


🚀 **Features**
✅ Search for any company by name or ticker  
✅ Fetch real-time stock price data using Alpha Vantage  
✅ Train an **LSTM model** with historical stock data with preferred no of days
✅ Choose from different forecast timelines (week/month/quarter)  
✅ Interactive price trend visualizations  


🛠 **Installation**

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Run the application
streamlit run app.py


🔑 API Key Configuration
This app uses Alpha Vantage API to fetch stock data.
Get your free API key from Alpha Vantage.

You can set your API key in Streamlit Secrets or manually enter it in the app.

📊 Technologies Used
Python
Streamlit (UI framework)
Pandas & NumPy (Data manipulation)
TensorFlow & Keras (LSTM model)
Plotly (Data visualization)
Alpha Vantage API (Stock data retrieval)


📌 Limitations
⚠️ The free Alpha Vantage API has a rate limit (5 requests per minute). If you hit the limit, wait a few minutes and try again.