from flask import Flask, jsonify, render_template
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market-overview', methods=['GET'])
def market_overview():
    try:
        # Fetch daily data for S&P 500 ETF (SPY)
        stock = yf.download('SPY', period='1mo', interval='1d')  # Adjust period and interval as needed

        # Check if the DataFrame is empty
        if stock.empty:
            return jsonify({"error": "No data retrieved from yfinance"}), 500

        # Extract 'Close' prices and timestamps
        prices = stock['Close'].values.tolist()  # Extract as numpy array and convert to list
        timestamps = stock.index.strftime('%Y-%m-%d').tolist()  # Format the index as dates

        return jsonify({"prices": prices, "timestamps": timestamps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

