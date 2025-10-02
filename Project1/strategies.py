import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

START_DATE = "2014-01-01"
END_DATE   = "2023-12-31"
TICKER     = "QQQ"
MONTHLY_INVEST = 500.0

def fetch_qqq_prices(start=START_DATE, end=END_DATE, save_csv=True):
    raw = web.DataReader(TICKER, "stooq", start, end).sort_index()

    df_csv = raw.reset_index().rename(columns={
        "Date": "date", "Open": "open", "Close": "close",
        "High": "high", "Low": "low", "Volume": "volume"
    })
    df_csv["symbol"] = TICKER
    df_csv = df_csv[["symbol","date","open","close","high","low","volume"]]
    df_csv["date"] = pd.to_datetime(df_csv["date"]).dt.strftime("%m-%d-%Y")
    if save_csv:
        df_csv.to_csv("QQQ.csv", index=False)

    return raw 

def to_monthly_close(daily_df):
    return daily_df["Close"].resample("M").last()

def ann_return_from_monthly_equity(eq):
    mret = eq.pct_change().dropna()
    if len(mret) == 0:
        return np.nan
    return (1 + mret.mean())**12 - 1

def total_return(ending_value, total_invested):
    return ending_value / total_invested - 1

def backtest_dca(ticker=TICKER, start=START_DATE, end=END_DATE, monthly_invest=MONTHLY_INVEST):
    prices = to_monthly_close(fetch_qqq_prices(start, end, save_csv=True))
    shares = 0.0
    equity = []
    contrib = []

   
    for p in prices:
        shares += monthly_invest / p          
        contrib.append(monthly_invest)
        equity.append(shares * p)

    eq = pd.Series(equity, index=prices.index)
    total_inv = np.cumsum(contrib)[-1]
    ann = ann_return_from_monthly_equity(eq)
    tot = total_return(eq.iloc[-1], total_inv)
    return ann, tot, eq  

def momentum_strategy(ticker=TICKER, start=START_DATE, end=END_DATE, monthly_invest=MONTHLY_INVEST):
    prices = to_monthly_close(fetch_qqq_prices(start, end, save_csv=False))
    monthly_returns = prices.pct_change()

    momentum = monthly_returns.rolling(window=6).apply(lambda x: x[:-1].sum())
    positions = (momentum > 0).astype(int) 
    cash, shares = 0.0, 0.0
    equity = []
    contrib = []

    for date, price in prices.items():
        cash += monthly_invest
        contrib.append(monthly_invest)

        if positions.loc[date] == 1 and cash > 0:
            shares += cash / price
            cash = 0.0

        equity.append(shares * price + cash)

    eq = pd.Series(equity, index=prices.index)
    total_inv = np.cumsum(contrib)[-1]
    ann = ann_return_from_monthly_equity(eq)
    tot = total_return(eq.iloc[-1], total_inv)
    return ann, tot, eq 

if __name__ == "__main__":
    dca_ann, dca_tot, dca_eq = backtest_dca()
    mom_ann, mom_tot, mom_eq = momentum_strategy()

    print("\nQQQ 2014–2023 (Stooq)")
    print(f"DCA:      Annualized Return = {dca_ann:6.2%} | Total Return = {dca_tot:6.2%}")
    print(f"Momentum: Annualized Return = {mom_ann:6.2%} | Total Return = {mom_tot:6.2%}")

    plt.figure(figsize=(10,6))
    dca_eq.plot(label=f"DCA (Ann {dca_ann:,.2%})")
    mom_eq.plot(label=f"Momentum (Ann {mom_ann:,.2%})")
    plt.title("QQQ – Strategy Equity Curves (Monthly, 2014–2023)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("qqq_strategy_comparison.png")
    plt.show()
