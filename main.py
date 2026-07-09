import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import psxdata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Ensure accurate Pakistan Standard Time
PKT = ZoneInfo("Asia/Karachi")

SECTORS = {
    "Banks": ["MEBL", "MCB", "UBL", "HBL", "BAHL"],
    "Fertilizer": ["EFERT", "FFC", "ENGRO", "FATIMA", "FFBL"],
    "Energy": ["HUBC", "OGDC", "PPL", "POL", "MARI"],
    "Cement": ["LUCK", "CHCC", "DGKC", "FCCL", "MLCF"],
    "Tech": ["SYS", "TRG", "PTC", "AVN", "NETSOL"]
}

TICKERS = [ticker for sector_list in SECTORS.values() for ticker in sector_list]

MARKET_CAPS = {
    "MEBL": 450.0, "MCB": 250.0, "UBL": 260.0, "HBL": 300.0, "BAHL": 100.0,
    "EFERT": 300.0, "FFC": 400.0, "ENGRO": 350.0, "FATIMA": 150.0, "FFBL": 50.0,
    "HUBC": 400.0, "OGDC": 550.0, "PPL": 350.0, "POL": 150.0, "MARI": 300.0,
    "LUCK": 350.0, "CHCC": 50.0, "DGKC": 80.0, "FCCL": 60.0, "MLCF": 50.0,
    "SYS": 250.0, "TRG": 100.0, "PTC": 60.0, "AVN": 20.0, "NETSOL": 15.0
}

def fetch_closing_prices(tickers, days=365):
    start_date = (datetime.now(PKT) - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for ticker in tickers:
        try:
            df = psxdata.stocks(ticker, start=start_date)
            if df is not None and not df.empty:
                data[ticker] = df['close']
        except Exception:
            pass
    portfolio_df = pd.DataFrame(data).ffill().dropna(axis=1)
    return portfolio_df

def black_litterman_optimization(prices_df, market_caps):
    # Guardrail: Abort if no data was fetched
    if prices_df.empty:
        print("Error: No price data available.")
        return {}

    returns = prices_df.pct_change().dropna()
    
    # Guardrail: Drop assets with zero variance to prevent LedoitWolf crashes
    returns = returns.loc[:, returns.var() > 0]
    
    tickers = returns.columns.tolist()
    num_assets = len(tickers)
    
    if num_assets == 0:
        return {}

    lw = LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_ * 252 
    
    caps = np.array([market_caps.get(t, 0) for t in tickers])
    if np.sum(caps) == 0:
        w_mkt = np.full(num_assets, 1.0 / num_assets)
    else:
        w_mkt = caps / np.sum(caps)
    
    risk_aversion = 2.5 
    pi = risk_aversion * np.dot(cov_matrix, w_mkt)
    
    P_list = []
    Q_list = []
    
    if "SYS" in tickers:
        sys_idx = tickers.index("SYS")
        row = np.zeros(num_assets)
        row[sys_idx] = 1.0
        P_list.append(row)
        Q_list.append(0.15) 
        
    if "EFERT" in tickers and "LUCK" in tickers:
        efert_idx = tickers.index("EFERT")
        luck_idx = tickers.index("LUCK")
        row = np.zeros(num_assets)
        row[efert_idx] = 1.0
        row[luck_idx] = -1.0
        P_list.append(row)
        Q_list.append(0.05) 
        
    if len(P_list) > 0:
        P = np.array(P_list)
        Q = np.array(Q_list)
        
        tau = 0.05
        omega = np.diag(np.diag(np.dot(np.dot(P, tau * cov_matrix), P.T)))
        
        tau_cov_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        term1 = np.linalg.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
        term2 = np.dot(tau_cov_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
        bl_returns = np.dot(term1, term2)
    else:
        bl_returns = pi 
    
    def objective_function(weights):
        port_return = np.dot(weights, bl_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        utility = port_return - (risk_aversion / 2) * port_variance
        return -utility 

    # Guardrail: Dynamic upper bounds prevent mathematically impossible constraints
    # If the API drops too many tickers, the ceiling automatically raises so the sum can reach 100%
    dynamic_ceiling = max(0.15, 1.0 / num_assets)
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0.01, dynamic_ceiling) for _ in range(num_assets)) 
    
    initial_guess = np.full(num_assets, 1.0 / num_assets) 
    
    result = minimize(
        objective_function, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return dict(zip(tickers, result.x))

if __name__ == "__main__":
    print(f"Fetching historical data for 25 PSX equities...")
    prices = fetch_closing_prices(TICKERS)
    
    print("Executing Black-Litterman Optimization...")
    optimal_weights = black_litterman_optimization(prices, MARKET_CAPS)
    
    if optimal_weights:
        print("\n--- Optimized Target Allocations ---")
        for ticker, weight in sorted(optimal_weights.items(), key=lambda item: item[1], reverse=True):
            print(f"{ticker:<6} | Target: {weight * 100:>5.2f}%")
