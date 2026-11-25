import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient


load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

# Stop early with a clear message if the keys didn't load
if api_key is None or secret_key is None:
    print("Keys not found. Check that .env is in the same folder and the names match.")
    exit()

# Connect to Alpaca's PAPER account (fake money)
client = TradingClient(api_key, secret_key, paper=True)

# Get the account info and print it
account = client.get_account()
print("Status:", account.status)
print("Cash:", account.cash)
print("Buying power:", account.buying_power)