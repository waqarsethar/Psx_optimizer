import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import psxdata
from datetime import datetime, timedelta

# Expanded Universe: Stratified across highly liquid KSE-30 constituents
SECTORS = {
    "Banks": ["MEBL", "MCB", "UBL"],
    "Fertilizer": ["EFERT", "FFC", "ENGRO"],
    "Energy": ["HUBC", "OGDC", "PPL"],
    "Cement": ["LUCK", "CHCC"],
    "Tech": ["SYS", "TRG"]
}

# Flatten the dictionary to a single ticker list
TICKERS = [ticker for sector_list in SECTORS.values() for ticker in sector_list]

def fetch_closing_prices(tickers, days=365):
    """Fetches real historical data."""
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for ticker in tickers:
        try:
            df = psxdata.stocks(ticker, start=start_date)
            if df is not None and not df.empty:
                data[ticker] = df['close']
        except Exception:
            pass
    portfolio_df = pd.DataFrame(data).ffill().dropna(axis=1) # Drop assets that failed to load
    return portfolio_df

def institutional_optimization(prices_df):
    """
    Applies Ledoit-Wolf Shrinkage and Constrained Optimization.
    """
    returns = prices_df.pct_change().dropna()
    num_assets = len(returns.columns)
    
    # 1. Ledoit-Wolf Covariance Shrinkage (Noise Reduction)
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(returns).covariance_ * 252 
    
    # Objective: Minimize Portfolio Variance
    def objective_variance(weights):
        return weights.T @ shrunk_cov_matrix @ weights

    # 2. Constraints: Sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # 3. Bounds: No stock > 15%, No stock < 2% (Prevents 0% and 52% allocations)
    # Adjust these limits based on your exact risk mandate
    bounds = tuple((0.02, 0.15) for _ in range(num_assets))
    
    # Initial equal-weight guess
    initial_guess = np.full(num_assets, 1.0 / num_assets)
    
    # SLSQP Solver execution
    result = minimize(
        objective_variance, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Return mapping of tickers to optimized weights
    return dict(zip(prices_df.columns, result.x))

if __name__ == "__main__":
    print("Executing Institutional Optimization with Ledoit-Wolf Shrinkage...")
    prices = fetch_closing_prices(TICKERS)
    
    if prices.empty:
        print("Failed to fetch historical data. Exiting.")
        exit()
        
    optimal_weights = institutional_optimization(prices)
    
    print("\n--- Constrained Target Allocations ---")
    for ticker, weight in sorted(optimal_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"{ticker:<6} | Target: {weight * 100:>5.2f}%")
