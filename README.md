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
Contribution levels tested ($100, $1,000, $5,000)
Number of dip thresholds tested (e.g., 3%, 5%, 10%, 15%)
Total number of simulations run

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
Assumptions (no trading fees, no slippage, dividents)

4. Results
Create a matrix with the results include Instrument, Contribution level, threshold level. The time window will be different as per data availability.

4.1 Portfolio Growth Over Time
Chart comparing total portfolio value
Commentary on growth patterns

4.2 Number & Timing of Purchases
How often each strategy invested
Periods of no investment for dip strategy
Months where dip strategy bought substantial amounts

4.3 Cash Accumulation Behavior
Maximum cash held
Max Length of time cash stayed idle

4.4 Returns Comparison
Total invested
Final value
Net return
Annualized return (CAGR)

4.5 Risk Metrics
Maximum cash idle time
Maximum drawdown
Volatility (std dev of monthly returns)
Sharpe ratio (risk-adjusted performance)

4.6 Sensitivity Analysis
Test dip thresholds: 3%, 5%, 10%, 15%
Show how results change with deeper dips
Show whether dip-buying is robust or fragile

5. Conclusion
Interpret the results:
When does DCA shine?
When does BTD work best?
Did BTD miss major bull runs?
Did cash accumulation reduce long-term compounding?
Did DCA buy at poor times too but still outperform?
Behavioral differences:
DCA = easy to follow
BTD = psychologically difficult (requires patience + discipline)

Summarize:
Which strategy performed better
Under which conditions
Key takeaways for investors
Whether dip-buying is worth the added complexity

6. Appendix
Full code
Raw transaction logs
