import streamlit as st
from transformers import pipeline
from openai import OpenAI
import requests
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries

# Initialize the question-answering pipeline
try:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased", framework="pt")
    print("Pipeline initialized successfully!")
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    qa_pipeline = None

# Set up Streamlit page config
st.set_page_config(page_title="Investing Learning App", layout="wide")

def home_page():
    st.title("📈 Welcome to the Finance learning App")
    st.write("""
        Your one-stop platform for financial education! Learn about:
        - 📊 Stock Market Investing
        - 💰 Cryptocurrency Trading
        - 🏖️ Planning for Retirement
        - 🤖 Interactive Financial Advice via Chatbot
    """)

    st.header("🚀 Latest Updates")
    # Fetch and display news articles
    api_key = st.secrets["newsapi"]["api_key"]  # Add this key to your secrets file
    articles = fetch_investing_news(api_key)

    if articles:
        for article in articles:
            st.subheader(article["title"])
            st.write(f"Published on: {article['publishedAt']}")
            st.write(article["description"])
            st.write(f"[Read more]({article['url']})")
            st.divider()
    else:
        st.info("No trending stories available at the moment. Check back later!")

    st.divider()
    st.header("🎯 Why Choose Us?")
    st.markdown("""
        - **User-Friendly:** Simplified tools and calculators.
        - **Comprehensive:** Covers diverse financial topics.
        - **Interactive:** Chatbot for tailored advice.
    """)
    st.subheader("📢 Get Started!")
    st.markdown("Begin by selecting a section from the sidebar to explore our offerings!")

    import requests

def fetch_investing_news(api_key, query="investing"):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return articles[:5]  # Return the top 5 articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Fetch stock data using Yahoo Finance (Primary) or Alpha Vantage (Fallback)
def fetch_stock_data(ticker):
    try:
        print(f"Fetching stock data for: {ticker} (Yahoo Finance)")
        data = yf.download(ticker, period="7d", interval="1h")
        if data.empty:
            raise ValueError(f"No data returned for ticker: {ticker} using Yahoo Finance")
        return data
    except Exception as e:
        print(f"Yahoo Finance error for {ticker}: {e}")

        # Fetch stock data using Alpha Vantage
        try:
            api_key = st.secrets["alpha_vantage"]["api_key"]  # Retrieve API key from secrets
            ts = TimeSeries(key=api_key, output_format="pandas")
            data, meta_data = ts.get_intraday(symbol=ticker, interval="60min", outputsize="full")
            data = data.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            data.index = pd.to_datetime(data.index)  # Ensure proper datetime index
            return data
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {e}")
            st.error(f"Failed to fetch data for {ticker} using both Yahoo Finance and Alpha Vantage.")
            return pd.DataFrame()

# Create candlestick chart
def plot_candlestick(data, ticker):
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )
    ])
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400
    )
    return fig


# Stock Market Page
def stock_market_page():
    st.title("📊 Stock Market Insights")

    # Allow user to select stocks
    st.subheader("📈 Live Stock Prices & Candlestick Charts")
    tickers = st.multiselect(
        "Select stocks to view:",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        default=["AAPL", "MSFT"]
    )
    
    if tickers:
        for ticker in tickers:
            st.write(f"### {ticker} Candlestick Chart")
            data = fetch_stock_data(ticker)
            
            if not data.empty:
                fig = plot_candlestick(data, ticker)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol or try again later.")

    st.subheader("🎥 How to Get Started in the Stock Market")
    st.video("https://youtu.be/p7HKvqRI_Bo?si=jZ3Xno3CaBaKQRsu")  # Example: Investing for Beginners
    st.video("https://youtu.be/ZCFkWDdmXG8?si=DnDGaE0UHkSlGtoB")  # Example: Stock Market Basics

    st.subheader("📚 Articles for Beginners")
    st.markdown("""
        - [Investopedia: How to Invest in Stocks](https://www.investopedia.com/how-to-invest-in-stocks-5076334)
        - [NerdWallet: A Beginner's Guide to Investing](https://www.nerdwallet.com/article/investing/stock-investing-guide)
        - [The Motley Fool: Stock Market Basics](https://www.fool.com/investing/how-to-invest/stocks/)
    """)


from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

import streamlit as st
import openai

def chatbot_page():
    st.title("🤖 Financial Advisor Chatbot")
    st.subheader("Ask Your Financial Questions")
    st.write("""
        Type your questions about stocks, cryptocurrencies, retirement, or general financial advice,
        and I'll provide insights using OpenAI's GPT model!
    """)

    # Retrieve OpenAI API key from Streamlit secrets
    openai_api_key = st.secrets["openai"]["api_key"]
    if not openai_api_key:
        st.error("OpenAI API key is missing. Please add it to your Streamlit secrets.")
        return

    # Set the API key for OpenAI
    openai.api_key = openai_api_key

    # Initialize the chat session if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for the user to send a message
    if prompt := st.chat_input("Ask your question here..."):

        # Save the user's message in session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the assistant's response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ],
                stream=True  # Enable streaming for partial responses
            )

            # Display the response as it streams and save it in session state
            assistant_message = ""
            with st.chat_message("assistant"):
                for chunk in response:
                    chunk_message = chunk["choices"][0].get("delta", {}).get("content", "")
                    assistant_message += chunk_message
                    st.markdown(assistant_message)
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

        except Exception as e:
            st.error(f"Error generating response: {e}")


def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Stock Market": stock_market_page,
        "Chatbot": chatbot_page,
    }
    choice = st.sidebar.selectbox("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()

