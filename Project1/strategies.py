import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

START_DATE = "2014-01-01"
END_DATE   = "2023-12-31"
TICKER     = "QQQ"
MONTHLY_INVEST = 500.0
RF_ANNUAL = 0.0 

def fetch_qqq_prices(start=START_DATE, end=END_DATE, save_csv=True):
    raw = web.DataReader(TICKER, "stooq", start, end).sort_index()

    df_csv = raw.reset_index().rename(columns={
        "Date": "date", "Open": "open", "Close": "close",
        "High": "high", "Low": "low", "Volume": "volume"
    })
    df_csv["symbol"] = TICKER
    df_csv = df_csv[["symbol","date","open","close","high","low","volume"]]
    df_csv = df_csv[
        (pd.to_datetime(df_csv["date"]) >= pd.to_datetime(start)) &
        (pd.to_datetime(df_csv["date"]) <= pd.to_datetime(end))
    ].copy()
    df_csv["date"] = pd.to_datetime(df_csv["date"]).dt.strftime("%m-%d-%Y")

    raw = raw.loc[pd.to_datetime(start):pd.to_datetime(end)].copy()
    return raw

def to_monthly_close(daily_df):
    monthly = daily_df["Close"].resample("M").last()
    monthly = monthly.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE)]
    return monthly

def ann_return_from_monthly_equity(eq):
    mret = eq.pct_change().dropna()
    years = (eq.index[-1].to_pydatetime() - eq.index[0].to_pydatetime()).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

def total_return(ending_value, total_invested):
    return ending_value / total_invested - 1

def ann_vol_from_equity(eq):
    mret = eq.pct_change().dropna()
    return mret.std(ddof=1) * np.sqrt(12)

def sharpe_from_equity(eq, rf_annual=RF_ANNUAL):
    mret = eq.pct_change().dropna()
    rf_m = (1 + rf_annual)**(1/12) - 1
    excess = mret - rf_m
    mu = excess.mean()
    sd = excess.std(ddof=1)
    return (mu * 12) / (sd * np.sqrt(12))

def drawdown_series(eq):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd, float(dd.min()) if len(dd) else np.nan

def backtest_dca(ticker=TICKER, start=START_DATE, end=END_DATE, monthly_invest=MONTHLY_INVEST):
    prices = to_monthly_close(fetch_qqq_prices(start, end, save_csv=True))
    shares = 0.0
    equity = []
    contrib = []

    for p in prices:
        shares += monthly_invest / p
        contrib.append(monthly_invest)
        equity.append(shares * p)

    eq = pd.Series(equity, index=prices.index, name="DCA_Equity")
    total_inv = float(np.cumsum(contrib)[-1])
    cagr = ann_return_from_monthly_equity(eq)
    tot = total_return(eq.iloc[-1], total_inv)
    vol = ann_vol_from_equity(eq)
    sharpe = sharpe_from_equity(eq, RF_ANNUAL)
    dd_curve, max_dd = drawdown_series(eq)

    return {
        "annualized_return": cagr,
        "total_return": tot,
        "vol_annual": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity": eq,
        "drawdown_curve": pd.Series(dd_curve, index=eq.index)
    }

def momentum_strategy(ticker=TICKER, start=START_DATE, end=END_DATE, monthly_invest=MONTHLY_INVEST):
    prices = to_monthly_close(fetch_qqq_prices(start, end, save_csv=False))
    monthly_returns = prices.pct_change()
    
    momentum = monthly_returns.shift(1).rolling(window=6).sum()
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

    eq = pd.Series(equity, index=prices.index, name="Momentum_Equity")
    total_inv = float(np.cumsum(contrib)[-1])
    cagr = ann_return_from_monthly_equity(eq)
    tot = total_return(eq.iloc[-1], total_inv)
    vol = ann_vol_from_equity(eq)
    sharpe = sharpe_from_equity(eq, RF_ANNUAL)
    dd_curve, max_dd = drawdown_series(eq)

    pct_in = float(positions.mean()) * 100.0

    return {
        "annualized_return": cagr,
        "total_return": tot,
        "vol_annual": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity": eq,
        "drawdown_curve": pd.Series(dd_curve, index=eq.index),
        "percent_invested": pct_in
    }

if __name__ == "__main__":
    dca = backtest_dca()
    mom = momentum_strategy()

    print("\nQQQ 2014–2023 (Stooq) — STRICT SPEC")
    print(f"DCA:      Annualized Return = {dca['annualized_return']:6.2%} | Total Return = {dca['total_return']:6.2%} | "
          f"Vol (Ann) = {dca['vol_annual']:5.2%} | Sharpe = {dca['sharpe']:.2f} | MaxDD = {dca['max_drawdown']:6.2%}")
    print(f"Momentum: Annualized Return = {mom['annualized_return']:6.2%} | Total Return = {mom['total_return']:6.2%} | "
          f"Vol (Ann) = {mom['vol_annual']:5.2%} | Sharpe = {mom['sharpe']:.2f} | MaxDD = {mom['max_drawdown']:6.2%} | "
          f"% Months Invested = {mom['percent_invested']:4.1f}%")
    
    plt.figure(figsize=(10,6))
    dca["equity"].plot(label=f"DCA (Ann {dca['annualized_return']:,.2%})")
    mom["equity"].plot(label=f"Momentum (Ann {mom['annualized_return']:,.2%})")
    plt.title("QQQ – Strategy Equity Curves (Monthly, 2014–2023)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("qqq_strategy_equity.png")
    plt.show()

    plt.figure(figsize=(10,6))
    dca["drawdown_curve"].plot(label="DCA")
    mom["drawdown_curve"].plot(label="Momentum")
    plt.title("QQQ – Strategy Drawdowns (2014–2023)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (fraction)")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("qqq_strategy_drawdowns.png")
    plt.show()

