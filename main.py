import pandas as pd
import numpy as np
from scipy.optimize import minimize
import psxdata
from datetime import datetime, timedelta
import requests
import os

TICKERS = ["MEBL", "EFERT", "HUBC", "LUCK", "SYS"]

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

def send_to_discord(message):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL environment variable is not set.")
    
    payload = {"content": message}
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()

if __name__ == "__main__":
    prices = fetch_closing_prices(TICKERS)
    optimal_weights = optimize_for_minimum_volatility(prices)
    
    # Format the message for Discord using Markdown code blocks for alignment
    report = f"📊 **Daily PSX Portfolio Targets** ({datetime.now().strftime('%Y-%m-%d')})\n```\n"
    
    for i, ticker in enumerate(TICKERS):
        live_data = psxdata.quote(ticker)
        current_price = live_data.get('current_price', 'N/A') if live_data is not None else 'N/A'
        weight_pct = optimal_weights[i] * 100
        report += f"{ticker:<6} | Target: {weight_pct:>5.2f}% | Live: Rs {current_price}\n"
        
    report += "```"
    send_to_discord(report)
