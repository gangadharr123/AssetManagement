# Pairs Trading Methodology Checklist

Based on "The Profitability of Pairs Trading Strategies: Distance, Cointegration, and Copula Methods" (Rad, Low & Faff, 2016).

## 1. DATA (Section 3)
- **Universe filters applied:** 
  - Daily data from the CRSP database (ordinary shares, share codes 10 and 11) traded on NYSE, AMEX, and NASDAQ. 
  - **Market Cap:** Exclude the bottom decile stocks in terms of market capitalization in each formation period.
  - **Price Floor:** Exclude stocks with prices less than $1.00 in the formation period.
  - **Volume Filter:** Exclude stocks that have at least one day without trading in any formation period in the respective trading period.
- **Exact formation period length:** 12 months.
- **Trading period length:** 6 months.
- **Monthly rolling:** The strategies are executed each month, without waiting 6 months for the current trading period to complete. This results in 6 overlapping "portfolios", with each portfolio associated with a trading period that has started in a different month.

## 2. DISTANCE METHOD (Section 4.1)
- **Normalized price:** Defined as the cumulative return index, adjusted for dividends and other corporate actions, and scaled to $1 at the beginning of the formation period ($P_{i,t}^{norm} = P_{i,t} / P_{i,0}$).
- **SSD Calculation:** The sum of squared differences (SSD) in their normalized prices during the formation period.
- **Pairs selected:** 20 pairs with the least SSD.
- **Exact opening threshold:** Spread diverges by 2 or more historical standard deviations (calculated in the formation period).
- **Closing condition:** The spread converges to zero.
- **Positions sizing:** $1 long-short positions (open a long position worth $1 and a short position worth $1 simultaneously).
- **Stock in multiple pairs:** Yes, a specific stock can participate in forming more than one pair as long as the other stock of the pair varies.
- **End of trading period:** If the spread hasn't converged by the end of the 6-month trading period, the positions are forced to close.
- **Return calculated:** Return is calculated as marked-to-market return on trades.

## 3. COINTEGRATION METHOD (Section 4.2)
- **2-step pair selection:** First, sort all possible combinations of pairs based on their SSD in their normalized price. Second, test each of the pairs with the least SSD for cointegration using the two-step Engle-Granger approach until 20 cointegrated pairs are selected.
- **Exact cointegration regression equation:**
  $X_{2,t} - \beta X_{1,t} = u_t$ (Equation 1)
  *(and its ECM representation: $X_{2,t} - X_{2,t-1} = \alpha_{X2}(X_{2,t-1} - \beta X_{1,t-1}) + \xi_{X2,t}$ )*
- **Spread formula (Equation 3):** 
  $spread_t = X_{2,t} - \beta X_{1,t}$
- **Normalized spread formula (Equation 6):** 
  $spread_{normalized} = \frac{spread - \mu_e}{\sigma_e}$
- **Opening/closing rule:** Open when the normalized spread diverges beyond 2 or -2. Close when the spread returns to zero.
- **Position sizing (beta-weighted):** If spread drops below -2, buy $1 worth of stock 2 and sell short $\beta$ dollars worth of stock 1. If spread moves above +2, sell short $1/\beta$ dollars worth of stock 2 and buy $1 worth of stock 1.
- **Profit formula (Equations 4-5):**
  Profit of trade at time t (Equation 4): $(X_{2,t} - X_{2,t-1}) - \beta(X_{1,t} - X_{1,t-1})$
  Rearranged (Equation 5): $(X_{2,t} - \beta X_{1,t}) - (X_{2,t-1} - \beta X_{1,t-1}) = spread_t - spread_{t-1}$

## 4. COPULA METHOD (Section 4.3)
- **Marginal distributions tried (4):** Extreme Value, Generalized Extreme Value, Normal, Logistic.
- **Copulas tried (5):** Clayton, Rotated Clayton, Gumbel, Rotated Gumbel, Student-t.
- **Model selection:** By maximizing the log likelihood of each copula density function and calculating the corresponding AIC and BIC. The copula/marginal associated with the highest AIC and BIC is selected.
- **Empirical results:** Student-t selected for ~62% (61.64%) of pairs; Logistic selected for ~86% (86.14%) of marginals.
- **Conditional probability formulas $h_1$ and $h_2$ (Equation 11):**
  $h_1(u_1|u_2) = P(U_1 \le u_1 | U_2 = u_2) = \frac{\partial C(u_1, u_2)}{\partial u_2}$
  $h_2(u_2|u_1) = P(U_2 \le u_2 | U_1 = u_1) = \frac{\partial C(u_1, u_2)}{\partial u_1}$
- **Mispricing indices $m_1, m_2$ (Equation 12):**
  $m_{1,t} = h_1(u_1|u_2) - 0.5 = P(U_1 \le u_1 | U_2 = u_2) - 0.5$
  $m_{2,t} = h_2(u_2|u_1) - 0.5 = P(U_2 \le u_2 | U_1 = u_1) - 0.5$
- **Cumulative indices $M_1, M_2$ (Equation 13):**
  $M_{1,t} = M_{1,t-1} + m_{1,t}$
  $M_{2,t} = M_{2,t-1} + m_{2,t}$
- **Exact opening threshold:** Open a long-short position once one of the cumulative mispriced indices is above +0.5 and the other one is below -0.5 at the same time.
- **Closing condition:** Unwound when both cumulative mispriced indices return to zero.

## 5. TRANSACTION COSTS (Section 4.4)
- **Cost model used:** Time-varying institutional commissions (from 70 bps down to 9 bps over 50 years) plus market impact estimates (30 bps for 1962-1988; 20 bps for 1989 onward). Short selling costs were excluded due to liquidity filters. Total costs were doubled for a complete pairs trade (two round-trip trades).
- **Adaptation for 2015-2025:** 5 bps commission, 20 bps market impact for modern S&P 500 (doubled for round trip).

## 6. PERFORMANCE CALCULATION (Section 4.5)
- **Return on Employed Capital formula (Equation 14):** 
  $REC_m = \frac{\sum_{i=1}^n r_i}{n}$
- **Return on Committed Capital formula (Equation 15):** 
  $RCC_m = \frac{\sum_{i=1}^{NP} r_i}{NP}$, where $NP = 20$.
- **6 overlapping portfolios:** Since formation and trading are done monthly (12-month formation, 6-month trading), there are 6 concurrent trading portfolios active in any given month.
- **Monthly excess return:** Computed as the equally weighted average return on these 6 overlapping portfolios.

## 7. EXPECTED RESULTS TO VALIDATE AGAINST
- **From Table 3 (After transaction costs, REC):**
  - **Distance:** Mean = 0.38% (0.0038), Sharpe = 0.3498
  - **Cointegration:** Mean = 0.33% (0.0033), Sharpe = 0.3497
  - **Copula:** Mean = 0.05% (0.0005), Sharpe = 0.0749
- **From Table 5 (Convergence rates):**
  - **Distance:** 62.53%
  - **Cointegration:** 61.35%
  - **Copula:** 39.98%
- **From Table 5 (Avg days open per converged trade):**
  - **Distance:** 21.15 days (mean)
  - **Cointegration:** 22.65 days (mean)
  - **Copula:** 26.30 days (mean)
- *Note:* 2015-2025 results will differ, but should produce similar patterns (DM ≈ Cointegration > Copula).
