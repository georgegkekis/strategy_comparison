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
import argparse
import glob
import os
import plotly.graph_objects as go

#   Example cases
#   S&P 500, 10y, DCA
#   S&P 500, 10y, 5% dip
#   S&P 500, 30y, 20% dip

CASES = [
{"inst":"S&P 500","ticker":"^GSPC","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Nasdaq 100","ticker":"^NDX","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Gold","ticker":"GLD","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Silver","ticker":"SLV","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Microsoft","ticker":"MSFT","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Intel","ticker":"INTC","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Tesla","ticker":"TSLA","years":[10,30],"dips":[None,0.05,0.10,0.20]},
{"inst":"Coca-Cola","ticker":"KO","years":[10,30],"dips":[None,0.05,0.10,0.20]},
]

def calculate_max_drawdown(series):
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()

def calculate_cagr(final_value, total_invested, start_date, end_date):
    days = (end_date - start_date).days
    years = days / 365.25
    if total_invested <= 0 or years <= 0:
        return np.nan
    return (final_value / total_invested) ** (1 / years) - 1

def parse_args():
    parser = argparse.ArgumentParser(description="DCA vs Buy-the-Dip Backtest")
    parser.add_argument("--ticker", type=str, default="^GSPC",
                        help="Instrument ticker (default: ^GSPC)")
    parser.add_argument("--monthly", type=float, default=1000.0,
                        help="Monthly contribution amount (default: 1000)")
    parser.add_argument("--start", type=str, default="2000-01-01",
                        help="Backtest start date YYYY-MM-DD (default: 2000-01-01)")
    parser.add_argument("--end", type=str, default="2025-01-01",
                        help="Backtest end date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Dip threshold as decimal (default: 0.05 = 5%%)")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run all tests in /tests directory")
    parser.add_argument("--run-batch", action="store_true",
                        help="Run all cases in CASES (batch mode)")

    return parser.parse_args()

def download_monthly_prices(ticker, start, end):
    print(f"Downloading {ticker} data from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker and network.")
    data['AdjClose'] = data['Adj Close']  # keep adjusted close
    # get month-end prices (last trading day of each month)
    monthly = data['AdjClose'].resample('M').last().to_frame(name='adj_close')
    monthly = monthly.dropna()
    monthly.index = monthly.index.tz_localize(None)
    print(f"Got {len(monthly)} month-ends. Range {monthly.index.min().date()} to {monthly.index.max().date()}")
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

def run_tests():
    print("Running tests...")

    test_dirs = glob.glob("tests/test*")
    if not test_dirs:
        print("No test directories found.")
        return

    for test_dir in test_dirs:
        print(f"\n=== Running test in {test_dir} ===")

        input_path = f"{test_dir}/input.csv"
        data_path = f"{test_dir}/data.csv"
        expected_path = f"{test_dir}/expected.csv"

        if not os.path.exists(input_path):
            print("Missing input.csv — skipping.")
            continue
        if not os.path.exists(data_path):
            print("Missing data.csv — skipping.")
            continue
        if not os.path.exists(expected_path):
            print("Missing expected.csv — skipping.")
            continue

        inp_df = pd.read_csv(input_path, header=None, names=["key", "value"])
        inp = inp_df.set_index("key")["value"].to_dict()

        monthly  = float(inp["monthly_amount"])
        th       = float(inp["dip_threshold"])

        monthly_prices = pd.read_csv(data_path, parse_dates=["date"])
        monthly_prices = monthly_prices.set_index("date")

        df_dca, shares_a, cash_a = simulate_dca(monthly_prices, monthly)
        df_dip, shares_b, cash_b = simulate_dip_strategy(monthly_prices, monthly, th)

        final_price = monthly_prices["adj_close"].iloc[-1]

        actual = {
            "final_value_dca": shares_a * final_price + cash_a,
            "shares_dca": shares_a,
            "cash_dca": cash_a,
            "final_value_dip": shares_b * final_price + cash_b,
            "shares_dip": shares_b,
            "cash_dip": cash_b
        }

        exp_df = pd.read_csv(expected_path, header=None, names=["key", "value"])
        expected = exp_df.set_index("key")["value"].astype(float).to_dict()

        for key in expected:
            e = expected[key]
            a = actual[key]
            assert np.isclose(a, e, rtol=1e-2, atol=1e-2), \
                f"{test_dir} | {key}: expected {e}, got {a}"

        print(f"{test_dir}: PASS")


def extract_years(start_date, end_date):
    d1 = pd.to_datetime(start_date)
    d2 = pd.to_datetime(end_date)
    years = (d2 - d1).days / 365.25
    return int(round(years))

def build_filename(ticker, years, dip, ext):
    dip_pct = int(dip * 100)
    safe_ticker = ticker.replace("^", "")
    return f"{safe_ticker}_{years}y_dca_vs_dip{dip_pct}.{ext}"

def plot_static_comparison(df_dca, df_dip, dip, ticker, years):
    title = f"{ticker} — {years} years"
    dip_pct = int(dip * 100)
    dip_label = f"Dip {dip_pct}%"

    perf = pd.DataFrame({
        "DCA": df_dca["total_value"],
        dip_label: df_dip["total_value"]
    })

    fname = build_filename(ticker, years, dip, "png")

    perf.plot(figsize=(10, 6), title=title)
    plt.ylabel("Portfolio value (USD)")
    plt.grid(True)
    plt.savefig(fname, dpi=250)
    plt.close()
    print(f"Static chart saved as {fname}")

def plot_interactive_comparison(df_dca, df_dip, dip, ticker, years):
    title = f"{ticker} — {years} years"

    dip_pct = int(dip * 100)
    dip_label = f"Dip {dip_pct}%"

    perf = pd.DataFrame({
        "DCA": df_dca["total_value"],
        dip_label: df_dip["total_value"]
    })

    fname = build_filename(ticker, years, dip, "html")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf["DCA"],
        mode="lines", name="DCA"
    ))

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf[dip_label],
        mode="lines", name=dip_label
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Portfolio value (USD)",
        xaxis_title="Date",
        hovermode="x unified"
    )

    fig.write_html(fname)
    print(f"Interactive chart saved as {fname}")

def longest_streak(no_buy_series):
    max_streak = 0
    current = 0
    for v in no_buy_series:
        if v == 1:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

def run_batch(args):
    end_dt = pd.to_datetime(args.end)
    summaries = []


    for case in CASES:
        inst = case["inst"]
        ticker = case["ticker"]
        safe_ticker = ticker.replace("^", "")

        for years in case["years"]:
            start_date = (end_dt - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

            for dip in case["dips"]:
                if dip is None:
                    continue

                print(f"\n== {inst} ({ticker}) | {years}y | dip {int(dip*100)}% ==")
                res = run_single(ticker, start_date, args.end, args.monthly, dip)

                res["inst"] = inst
                summaries.append(res)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("summary.csv", index=False)

def run_single(ticker, start_date, end_date, monthly_amount, drop_threshold):
    TICKER = ticker
    START_DATE = start_date
    END_DATE = end_date
    MONTHLY_AMOUNT = monthly_amount
    DROP_THRESHOLD = drop_threshold

    monthly = download_monthly_prices(TICKER, START_DATE, END_DATE)
    start_dt = monthly.index[0].to_pydatetime()
    end_dt   = monthly.index[-1].to_pydatetime()

    # Strategy A: DCA
    df_dca, shares_a, cash_a = simulate_dca(monthly, MONTHLY_AMOUNT)
    final_price = monthly['adj_close'].iloc[-1]
    final_value_a = shares_a * final_price + cash_a
    total_invested_a = MONTHLY_AMOUNT * len(df_dca)
    net_return_a = (final_value_a - total_invested_a) / total_invested_a
    max_dd_a = calculate_max_drawdown(df_dca["total_value"])
    cagr_a = calculate_cagr(final_value_a, total_invested_a, start_dt, end_dt)

    # Strategy B: buy-the-dip
    df_dip, shares_b, cash_b = simulate_dip_strategy(monthly, MONTHLY_AMOUNT, DROP_THRESHOLD)
    final_value_b = shares_b * final_price + cash_b
    total_invested_b = MONTHLY_AMOUNT * len(df_dip)
    net_return_b = (final_value_b - total_invested_b) / total_invested_b
    dip_invest_months = (df_dip["shares_bought"] > 0).sum()
    dip_no_invest_months = df_dip.index[df_dip["shares_bought"] == 0]
    # Longest consecutive waiting streak (in months)
    no_buy = (df_dip["shares_bought"] == 0).astype(int)
    max_dd_b = calculate_max_drawdown(df_dip["total_value"])
    invested_b = MONTHLY_AMOUNT * len(monthly) - df_dip['cash'].iloc[-1]
    cagr_b = calculate_cagr(final_value_b, total_invested_b, start_dt, end_dt)
    max_cash_held = df_dip["cash"].max()

    max_wait = longest_streak(no_buy)

    substantial_buys = df_dip[df_dip["shares_bought"] * df_dip["price"] >= 12 * MONTHLY_AMOUNT]
    substantial_buys = substantial_buys.copy()  # avoid SettingWithCopyWarning
    substantial_buys["invested"] = substantial_buys["shares_bought"] * substantial_buys["price"]

    # Print summary
    print("\n--- Summary ---")
    print(f"Months simulated: {len(monthly)}")
    print(f"Final price (last month-end): {final_price:.2f}")
    print("\nStrategy A (DCA):")

    print(f"  DCA net return: {net_return_a*100:.2f}%")
    print(f"  CAGR DCA: {cagr_a*100:.2f}%")
    print(f"  Total invested: ${total_invested_a:,.2f}")
    print(f"  Final portfolio value: ${final_value_a:,.2f}")
    print(f"  Max drawdown: {max_dd_a*100:.2f}%")
    print(f"  Shares held: {shares_a:.6f}  Cash leftover: ${cash_a:.2f}")

    print("\nStrategy B (Buy-the-dip):")
    print(f"  DIP net return: {net_return_b*100:.2f}%")
    print(f"  CAGR DIP: {cagr_b*100:.2f}%")
    print(f"  Total money supplied (budgeted): ${MONTHLY_AMOUNT * len(monthly):,.2f}")
    print(f"  Total invested (actually spent): ${invested_b:,.2f}")
    print(f"  Final portfolio value: ${final_value_b:,.2f}")
    print(f"  Shares held: {shares_b:.6f}  Cash leftover: ${cash_b:.2f}")
    print(f"  Max consecutive months waiting for dip: {max_wait}")
    print(f"  Maximum cash held: ${max_cash_held:,.2f}")
    print(f"  Max drawdown: {max_dd_b*100:.2f}%")

    print("\n--- Additional Analysis ---")
    print(f"DIP invested in {dip_invest_months} months")
    print(f"DIP skipped {len(dip_no_invest_months)} months")
    print("\nSubstantial DIP buys:")
    print (f"{substantial_buys[['price','shares_bought','invested']]}"
    if not substantial_buys.empty else "\nNo substantial DIP buys")

    # Save results and plots
    df_dca.to_csv("strategy_dca_history.csv")
    df_dip.to_csv("strategy_dip_history.csv")
    print("\nDetailed histories saved to strategy_dca_history.csv and strategy_dip_history.csv")

    years = extract_years(START_DATE, END_DATE)
    plot_static_comparison(df_dca, df_dip, DROP_THRESHOLD, TICKER, years)
    plot_interactive_comparison(df_dca, df_dip, DROP_THRESHOLD, TICKER, years)

    return {
        "ticker": TICKER,
        "start": START_DATE,
        "end": END_DATE,
        "years": years,
        "dip": DROP_THRESHOLD,

        "dca_net_return": net_return_a,
        "dip_net_return": net_return_b,
        "net_return_diff": net_return_b - net_return_a,  # + means DIP beat DCA

        "winner": "DIP" if net_return_b > net_return_a else "DCA",

        "dca_final_value": final_value_a,
        "dip_final_value": final_value_b,
        "dca_cagr": cagr_a,
        "dip_cagr": cagr_b,
        "dca_max_dd": max_dd_a,
        "dip_max_dd": max_dd_b,
        "dip_max_wait_months": max_wait,
    }

def main():
    args = parse_args()
    if args.run_tests:
        run_tests()
        return
    if args.run_batch:
        run_batch(args)
        return
    run_single(args.ticker, args.start, args.end, args.monthly, args.threshold)

if __name__ == "__main__":
    main()

