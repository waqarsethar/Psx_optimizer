import os
import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import psxdata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TICKERS = ["MEBL", "EFERT", "HUBC", "LUCK", "SYS"]
PKT = ZoneInfo("Asia/Karachi")

SECTOR_MAP = {
    "MEBL": "🏦 Commercial Banks",
    "EFERT": "🌱 Fertilizer",
    "HUBC": "⚡ Power Generation",
    "LUCK": "🏗️ Cement",
    "SYS":  "💻 Technology"
}

def fetch_closing_prices(tickers, days=365):
    # Ensure start date is calculated based on PKT
    start_date = (datetime.now(PKT) - timedelta(days=days)).strftime("%Y-%m-%d")
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

def extract_live_price(ticker):
    """Bulletproof extraction handling Pandas objects, changing headers, and string formatting."""
    try:
        live_data = psxdata.quote(ticker)
        if live_data is None:
            return "N/A"
            
        # Standardize live_data into a flat dictionary with lowercase keys
        if isinstance(live_data, pd.DataFrame):
            if live_data.empty: return "N/A"
            data_dict = {str(k).lower(): v for k, v in live_data.iloc[0].to_dict().items()}
        else:
            data_dict = {str(k).lower(): v for k, v in dict(live_data).items()}
            
        # Check against common PSX column names
        for key in ['current', 'price', 'close', 'last', 'ldcp']:
            if key in data_dict and pd.notna(data_dict[key]):
                # Convert to string, strip commas (e.g., "1,200.50" -> "1200.50"), then float
                raw_val = str(data_dict[key]).replace(',', '').strip()
                return float(raw_val)
                
    except Exception as e:
        print(f"Error parsing live price for {ticker}: {e}")
        
    return "N/A"

def send_discord_embed(embed_dict):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL is not set.")
        return
    
    payload = {"embeds": [embed_dict]}
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()

if __name__ == "__main__":
    prices = fetch_closing_prices(TICKERS)
    optimal_weights = optimize_for_minimum_volatility(prices)
    
    embed_fields = []
    for i, ticker in enumerate(TICKERS):
        current_price = extract_live_price(ticker)
        weight_pct = optimal_weights[i] * 100
        sector_label = SECTOR_MAP.get(ticker, "Sector")
        
        field_value = f"**Target:** {weight_pct:.2f}%\n\n**Live:** Rs {current_price}"
        
        embed_fields.append({
            "name": f"{sector_label} ({ticker})",
            "value": field_value,
            "inline": True
        })
        
    embed = {
        "title": "📈 PSX Daily Portfolio Optimization",
        "description": "Minimum volatility target allocations based on 1-year covariance.",
        "color": 3066993, 
        "fields": embed_fields,
        "footer": {
            # Use the exact PKT timestamp instead of server UTC
            "text": f"Generated: {datetime.now(PKT).strftime('%Y-%m-%d %H:%M:%S PKT')}"
        }
    }
    
    send_discord_embed(embed)
