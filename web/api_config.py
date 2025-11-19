# Live Data API Configuration
# Add your free API keys here to get real live stock data

# Finnhub (PRIMARY - Best for unlimited calls)
# Get free key: https://finnhub.io/register  
# Limits: 60 calls/minute, no daily limit on free tier
FINNHUB_API_KEY = "d465hppr01qj716ff080d465hppr01qj716ff08g"  # Replace with your actual key

# Alpha Vantage (BACKUP - Use when Finnhub fails)
# Get free key: https://www.alphavantage.co/support/#api-key
# Limits: 25 calls/day, 5 calls/minute
ALPHA_VANTAGE_API_KEY = "4MGMG0UKR7X9IMGB"

# Polygon.io (Optional)
# Get free key: https://polygon.io/
POLYGON_API_KEY = "demo"

# Instructions:
# 1. Sign up for FREE Finnhub account: https://finnhub.io/register
# 2. Replace "demo" with your actual Finnhub API key
# 3. Restart the Flask app
# 4. Enjoy 60 calls/minute instead of 25 calls/day!

# Priority Order:
# 1. Finnhub (60 calls/minute) 
# 2. Alpha Vantage (25 calls/day)
# 3. Demo data (fallback)