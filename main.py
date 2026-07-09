import os
import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import psxdata
from datetime import datetime, timedelta

TICKERS = ["MEBL", "EFERT", "HUBC", "LUCK", "SYS"]

# Map tickers to sectors for the embed fields
SECTOR_MAP = {
    "MEBL": "🏦 Commercial Banks",
    "EFERT": "🌱 Fertilizer",
    "HUBC": "⚡ Power Generation",
    "LUCK": "🏗️ Cement",
    "SYS":  "💻 Technology"
}

def fetch_closing_prices(tickers, days=365):
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for ticker in tickers:
        try:
            df = psxdata.stocks(ticker, start=start_date)
            data[ticker] = df['close']
        except Exception:
            pass
    portfolio_df = pd.DataFrame(data)
    portfolio_df.ffill(inplace=True)
    return portfolio_df

def optimize_for_minimum_volatility(prices_df):
    returns = prices_df.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    num_assets = len(returns.columns)
    
    def objective_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.full(num_assets, 1.0 / num_assets)
    
    result = minimize(objective_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def send_discord_embed(embed_dict):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL environment variable is not set.")
    
    # Discord expects an 'embeds' array in the JSON payload
    payload = {"embeds": [embed_dict]}
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()

if __name__ == "__main__":
    prices = fetch_closing_prices(TICKERS)
    optimal_weights = optimize_for_minimum_volatility(prices)
    
    # 1. Build the individual fields for each sector/ticker
    embed_fields = []
    for i, ticker in enumerate(TICKERS):
        live_data = psxdata.quote(ticker)
        current_price = live_data.get('current_price', 'N/A') if live_data is not None else 'N/A'
        weight_pct = optimal_weights[i] * 100
        
        sector_label = SECTOR_MAP.get(ticker, "Sector")
        
        embed_fields.append({
            "name": f"{sector_label} ({ticker})",
            "value": f"**Target:** {weight_pct:.2f}%\n**Live:** Rs {current_price}",
            "inline": True # Set to False if you want them stacked vertically
        })
        
    # 2. Construct the main embed object
    embed = {
        "title": "📈 PSX Daily Portfolio Optimization",
        "description": "Minimum volatility target allocations based on 1-year covariance.",
        "color": 3066993, # Hex 0x2ECC71 (Emerald Green) converted to a base-10 integer
        "fields": embed_fields,
        "footer": {
            "text": f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S PKT')}"
        }
    }
    
    # 3. Dispatch the payload
    send_discord_embed(embed)
