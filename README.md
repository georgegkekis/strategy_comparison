# strategy_comparison

# DCA vs. Buy-the-Dip: A Backtest Study

Skeleton
1. Abstract
Purpose of the study
Methods (historical backtest, instruments, monthly contributions, dip threshold)
Key findings (The general conclusions)
Implications for investors

2. Introduction
Define Dollar-Cost Averaging (DCA)
Define Buy-the-Dip (BTD) timing strategy
Why these two strategies are interesting to compare

3. Methodology
3.1 Data Source
Instruments: List ETFs/stocks used
Time Period: 1990–present (or available data)
Data Provider: yfinance
Price Type: Adjusted close
Frequency:

3.2 Strategy Definitions
DCA Strategy:
Buy-the-Dip Strategy:

3.3 Backtest Logic
Resample daily data to month-end
Track for each month:
    Price
    Cash balance
    Shares held
    Total portfolio value
    All-time-high
Assumptions (no trading fees, no slippage, dividents)

3.4 Contribution levels

3.5 Dip Thresholds
Vary dip thresholds (3%, 5%, 10%, 15%)
Test different periods within 1990–present to capture multiple market regimes

3.6 Limitations
Past performance ≠ future performance

4. Results
5. Conclusion
6. Appendix
