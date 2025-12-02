#!/usr/bin/env python3

"""
Backtest comparison: Dollar-Cost Averaging (DCA) vs. "Buy the Dip" timing strategy.

This script evaluates two investment approaches over a historical price period:

STRATEGY A — **DCA (Dollar-Cost Averaging)**
    Invest a fixed amount (e.g., $1000) at the same time each month,
    regardless of market conditions.
    - Always buys at the chosen monthly date (month-end by default).
    - Represents a fully mechanical, non-timed investing strategy.

STRATEGY B — **Buy the Dip**
    Budget the same monthly amount, but only invest when the asset has fallen
    by at least a specified percentage from its all-time high (e.g., 5%+ drawdown).
    - Unused monthly contributions accumulate as cash.
    - When a qualifying dip occurs, all accumulated cash is invested at once.
    - Represents a simple timing strategy based on drawdowns.

WHAT THE SCRIPT DOES
     Downloads historical prices using yfinance.
     Converts daily prices to month-end prices for consistent monthly decision points.
     Simulates both strategies over identical periods and price data.
     Tracks shares, cash, portfolio value, and investment timing.
     Saves full transaction histories and a comparison plot.

WHAT THE SCRIPT SEEKS TO ANSWER
    “Over this historical period, did systematic dip-buying outperform
    regular DCA— or vice-versa?”

LIMITATIONS
     Past performance does not guarantee future results.
     This is not a predictive model, only a historical backtest.

Use this script to test whether a “wait for dips” approach improves,
hurts, or matches performance compared to pure DCA under real historical data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

START_DATE = "2003-01-01"
END_DATE = "2025-01-01"
TICKER = "SPY"
MONTHLY_AMOUNT = 1000.0
DROP_THRESHOLD = 0.05     # 5% drop threshold (5% = 0.05)

def download_monthly_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker and network.")
    data['AdjClose'] = data['Adj Close']  # keep adjusted close
    # get month-end prices (last trading day of each month)
    monthly = data['AdjClose'].resample('M').last().to_frame(name='adj_close')
    monthly = monthly.dropna()
    monthly.index = monthly.index.tz_localize(None)
    return monthly

def simulate_dca(monthly_prices, monthly_amount):
    shares = 0.0
    cash = 0.0
    history = []
    for dt, row in monthly_prices.iterrows():
        price = float(row['adj_close'])
        shares_bought = monthly_amount / price
        shares += shares_bought
        cash += 0.0
        total_value = shares * price + cash
        history.append({
            'date': dt.date(),
            'action': 'buy',
            'cash_in': monthly_amount,
            'shares_bought': shares_bought,
            'price': price,
            'shares': shares,
            'cash': cash,
            'total_value': total_value
        })
    df = pd.DataFrame(history).set_index('date')
    return df, shares, cash

def simulate_dip_strategy(monthly_prices, monthly_amount, drop_threshold):
    shares = 0.0
    cash = 0.0
    all_time_high = -np.inf
    history = []

    for dt, row in monthly_prices.iterrows():
        price = float(row['adj_close'])
        cash += monthly_amount
        action = 'hold'
        shares_bought = 0.0

        # Update all-time high
        if price > all_time_high:
            all_time_high = price

        if price <= all_time_high * (1 - drop_threshold):
            shares_bought = cash / price
            shares += shares_bought
            action = 'buy_on_dip'
            cash = 0.0

        total_value = shares * price + cash
        history.append({
            'date': dt.date(),
            'action': action,
            'cash_in': monthly_amount,
            'shares_bought': shares_bought,
            'price': price,
            'shares': shares,
            'cash': cash,
            'all_time_high': all_time_high,
            'total_value': total_value
        })

    df = pd.DataFrame(history).set_index('date')
    return df, shares, cash


def main():
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
    monthly = download_monthly_prices(TICKER, START_DATE, END_DATE)
    print(f"Got {len(monthly)} month-ends. Range {monthly.index.min().date()} to {monthly.index.max().date()}")

    # Strategy A: DCA
    df_a, shares_a, cash_a = simulate_dca(monthly, MONTHLY_AMOUNT)
    final_price = monthly['adj_close'].iloc[-1]
    final_value_a = shares_a * final_price + cash_a
    total_invested_a = MONTHLY_AMOUNT * len(df_a)

    # Strategy B: only buy on >=5% down vs previous month
    df_b, shares_b, cash_b = simulate_dip_strategy(monthly, MONTHLY_AMOUNT, DROP_THRESHOLD)
    final_value_b = shares_b * final_price + cash_b
    total_invested_b = MONTHLY_AMOUNT * len(df_b) - df_b['cash'].iloc[-1]  # approx: invested = total supplied - leftover cash

    # Print summary
    print("\n--- Summary ---")
    print(f"Months simulated: {len(monthly)}")
    print(f"Final price (last month-end): {final_price:.2f}")
    print("\nStrategy A (DCA):")
    print(f"  Total invested: ${total_invested_a:,.2f}")
    print(f"  Final portfolio value: ${final_value_a:,.2f}")
    print(f"  Shares held: {shares_a:.6f}  Cash leftover: ${cash_a:.2f}")

    print("\nStrategy B (Buy on >=5% monthly dip):")
    invested_b = MONTHLY_AMOUNT * len(monthly) - df_b['cash'].iloc[-1]
    print(f"  Total money supplied (budgeted): ${MONTHLY_AMOUNT * len(monthly):,.2f}")
    print(f"  Total invested (actually spent): ${invested_b:,.2f}")
    print(f"  Final portfolio value: ${final_value_b:,.2f}")
    print(f"  Shares held: {shares_b:.6f}  Cash leftover: ${cash_b:.2f}")

    # Save results and plots
    df_a.to_csv("strategy_dca_history.csv")
    df_b.to_csv("strategy_dip_history.csv")
    print("\nDetailed histories saved to strategy_dca_history.csv and strategy_dip_history.csv")

    # Combined performance plot
    perf = pd.DataFrame({
        'DCA_value': df_a['total_value'],
        'Dip_value': df_b['total_value']
    })
    perf.plot(figsize=(10,6), title="Portfolio value over time")
    plt.ylabel("Portfolio value (USD)")
    plt.grid(True)
    plt.savefig("strategy_comparison.png", dpi=150)
    print("Comparison chart saved as strategy_comparison.png")

if __name__ == "__main__":
    main()

