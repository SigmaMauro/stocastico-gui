# -*- coding: utf-8 -*-
"""
Backtest "brevissimo" vs "medio" (split)
----------------------------------------
- Calcola DUE segnali as-of per ogni giorno di trading:
    * BREVE (1–2g)  → usato per d1–d2
    * MEDIO (3–15g) → usato per d3–d5
- Anchor = Close DAILY del giorno del segnale (stesso anchor per breve/medio)
- Esito per ciascun giorno X:
    LONG  → High_X > anchor
    SHORT → Low_X  < anchor
- Salta weekend/festivi usando i trading days dal daily.

Esecuzione (con o senza argomenti):
  python backtest_breve_medio_split.py --tickers GOOGL MSFT --days-back 60 --threshold 0.30 --debug

Requisiti:
  pip install yfinance pandas numpy
  Il file 'stocastico_gui.py' deve stare nella stessa cartella (riusa stochastic/make_recommendations).
"""
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Importa le tue funzioni (stochastic, make_recommendations) dal progetto
import stocastico_gui as sg


# ======================== CONFIG DI DEFAULT (modifica liberamente) ========================
DEFAULT_TICKERS = ["googl", "msft","v","ma","mrna","pfe","spot","nflx"]
DEFAULT_DAYS_BACK = 20
DEFAULT_THRESHOLD = 0.30
# =========================================================================================


# ---------------------- Preset scenari/indicatori (coerenti con la tua GUI) --------------
BREVI_SCENARIOS: List[Tuple[int, str, float]] = [
    (2,  "5m",  0.10),
    (5,  "15m", 0.20),
    (10, "30m", 0.40),
    (10, "1h",  0.25),
    (20, "1h",  0.05),
]
BREVI_IND_WEIGHTS: Dict[str, float] = {
    "Stoch": 0.40,
    "MACD": 0.20,
    "MACD_hist": 0.10,
    "RSI": 0.15,
    "Trend": 0.10,
    "OBV": 0.05,
}

MEDIO_SCENARIOS: List[Tuple[int, str, float]] = [
    (3,  "30m", 0.20),
    (5,  "1h",  0.25),
    (10, "90m", 0.30),
    (15, "4h",  0.15),
    (30, "4h",  0.10),
]
MEDIO_IND_WEIGHTS: Dict[str, float] = {
    "Stoch": 0.15,
    "RSI": 0.15,
    "MACD": 0.25,
    "MACD_hist": 0.15,
    "OBV": 0.15,
    "Trend": 0.15,
}

VOTE_VAL: Dict[str, int] = {"long": 1, "short": -1, "neutral": 0}


# ---------------------- Utility ----------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rende le colonne standard: Open/High/Low/Close/Volume anche se yfinance ritorna MultiIndex."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [''.join([str(x) for x in col if str(x) != '']) for col in df.columns.values]
        for base in ["Open", "High", "Low", "Close", "Volume"]:
            matches = [c for c in df.columns if c.startswith(base)]
            if matches:
                df[base] = df[matches[0]]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if keep:
            df = df[keep]
    return df


def bars_per_day(interval: str) -> float:
    mapping = {
        "1d": 1.0, "1h": 7.0, "90m": 4.5, "60m": 7.0, "30m": 13.0,
        "15m": 26.0, "5m": 78.0, "1m": 390.0, "4h": 1.8
    }
    return mapping.get(interval, 1.0)


def recommended_k(days: int, interval: str, cap_bars: Optional[int] = None) -> int:
    bars = int(round(days * bars_per_day(interval)))
    if cap_bars is not None:
        bars = min(bars, cap_bars)
    k = max(5, min(300, int(round(bars * 0.45))))
    k = max(5, min(k, max(5, bars - 2)))
    return k


def yahoo_cap_period(days: int, interval: str) -> str:
    if interval in ("1m",):
        return f"{min(days, 7)}d"
    elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h", "2h", "4h", "6h"):
        return f"{min(days, 60)}d"
    else:
        return f"{days}d"


def fetch_data_asof(ticker: str, days: int, interval: str, asof_dt: pd.Timestamp,
                    pre_buffer_days: int = 240) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Scarica abbastanza storico per calcolare indicatori **fino ad asof_dt**."""
    total_days = days + pre_buffer_days + 7
    period_all = yahoo_cap_period(total_days, interval)
    df_all = yf.download(ticker, period=period_all, interval=interval, auto_adjust=False, progress=False)
    df_all = normalize_ohlcv(df_all)
    if df_all is None or df_all.empty:
        raise RuntimeError("Dati non disponibili")
    if getattr(df_all.index, "tz", None) is not None:
        df_all.index = df_all.index.tz_localize(None)
    df_all = df_all.sort_index()

    # taglia al giorno del segnale (asof_dt 23:59)
    day_end = pd.Timestamp(asof_dt.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff_idx = df_all.index[df_all.index <= day_end]
    if len(cutoff_idx) == 0:
        fallback_limit = day_end + pd.Timedelta(hours=12)
        cutoff_idx = df_all.index[df_all.index <= fallback_limit]
        if len(cutoff_idx) == 0:
            raise RuntimeError("Nessuna barra entro la data As-Of.")
    last_asof_ts = cutoff_idx[-1]

    df_past = df_all.loc[:last_asof_ts].tail(days + pre_buffer_days)
    if df_past.shape[0] < 60:
        raise RuntimeError("Storico insufficiente prima della data As-Of.")
    return df_past, last_asof_ts


def scenario_score(df: pd.DataFrame, last_asof_ts: pd.Timestamp,
                   days: int, interval: str, ind_weights: Dict[str, float]) -> float:
    k_auto = recommended_k(days, interval)
    # usa i tuoi indicatori
    _, sk, sd = sg.stochastic(df, k_period=k_auto, smooth_k=3, smooth_d=3)
    sk = sk.loc[:last_asof_ts]
    sd = sd.loc[:last_asof_ts]
    rec = sg.make_recommendations(df.loc[:last_asof_ts], sk, sd)
    sig = rec.get("signals", {}) or {}
    scen = 0.0
    for ind, w in ind_weights.items():
        scen += VOTE_VAL.get(sig.get(ind, "neutral"), 0) * w
    return scen


def weighted_check_asof(ticker: str, asof_dt: pd.Timestamp,
                        scenarios: List[Tuple[int, str, float]],
                        ind_weights: Dict[str, float]) -> float:
    total = 0.0
    for days, interval, scen_w in scenarios:
        df_past, last_ts = fetch_data_asof(ticker, days, interval, asof_dt)
        s = scenario_score(df_past, last_ts, days, interval, ind_weights)
        total += s * scen_w
    return float(total)


def outcome_from_daily(daily_df: pd.DataFrame,
                       anchor_close: float,
                       days_after: List[pd.Timestamp],
                       side: str) -> List[str]:
    """Valuta gli esiti su giorni di TRADING (daily)."""
    esiti: List[str] = []
    for d in days_after:
        if d not in daily_df.index:
            esiti.append("NO")
            continue
        row = daily_df.loc[d]
        try:
            hi = float(row["High"])
            lo = float(row["Low"])
        except KeyError:
            esiti.append("NO")
            continue
        if np.isnan(hi) or np.isnan(lo):
            esiti.append("NO")
            continue
        if side == "LONG":
            esiti.append("SI" if hi > anchor_close else "NO")
        elif side == "SHORT":
            esiti.append("SI" if lo < anchor_close else "NO")
        else:
            esiti.append("NA")
    return esiti


# ---------------------- Backtest ----------------------
@dataclass
class BacktestRow:
    date: str
    action_b: str
    score_b: float
    action_m: str
    score_m: float
    d1: str; d2: str; d3: str; d4: str; d5: str


def run_for_ticker(ticker: str, days_back: int, threshold: float, debug: bool = False) -> pd.DataFrame:
    rows: List[BacktestRow] = []
    today = dt.date.today()
    start_date = today - dt.timedelta(days=days_back)
    end_date = today - dt.timedelta(days=7)  # fino a una settimana fa

    # DAILY per anchor + outcome
    day_period = f"{days_back + 30}d"
    df_daily = yf.download(ticker, period=day_period, interval="1d", auto_adjust=False, progress=False)
    df_daily = normalize_ohlcv(df_daily)
    if df_daily is None or df_daily.empty:
        raise RuntimeError(f"Nessun daily per {ticker}")
    if getattr(df_daily.index, "tz", None) is not None:
        df_daily.index = df_daily.index.tz_localize(None)
    df_daily = df_daily.sort_index()

    trade_days = [d for d in df_daily.index if (start_date <= d.date() <= end_date)]
    print(f"[{ticker}] trading days da testare: {len(trade_days)} (da {start_date} a {end_date})")
    idx_list = list(df_daily.index)

    for d0 in trade_days:
        anchor_close = float(df_daily.at[d0, "Close"])
        asof_ts = pd.Timestamp(d0) + pd.Timedelta(hours=23, minutes=59)

        # Brevissimo
        try:
            score_b = weighted_check_asof(ticker, asof_ts, BREVI_SCENARIOS, BREVI_IND_WEIGHTS)
        except Exception as e:
            if debug:
                print(f"  - {d0.date()} SKIP (errore calc breve: {e})")
            continue
        action_b = "LONG" if score_b >= threshold else ("SHORT" if score_b <= -threshold else "NA")

        # Medio
        try:
            score_m = weighted_check_asof(ticker, asof_ts, MEDIO_SCENARIOS, MEDIO_IND_WEIGHTS)
        except Exception as e:
            if debug:
                print(f"  - {d0.date()} SKIP (errore calc medio: {e})")
            continue
        action_m = "LONG" if score_m >= threshold else ("SHORT" if score_m <= -threshold else "NA")

        # prossimi 5 trading days
        try:
            pos0 = idx_list.index(d0)
        except ValueError:
            if debug:
                print(f"  - {d0.date()} SKIP (giorno non in index)")
            continue
        next_days = [idx_list[pos0 + k] for k in (1, 2, 3, 4, 5) if (pos0 + k) < len(idx_list)]

        if debug:
            hl = []
            for x in next_days:
                hi = float(df_daily.at[x, "High"])
                lo = float(df_daily.at[x, "Low"])
                hl.append((str(x.date()), hi, lo))
            print(f"  {d0.date()} | anchor={anchor_close:.2f}  B:({action_b},{score_b:.3f})  M:({action_m},{score_m:.3f})")
            print(f"    next days H/L: {hl}")

        # d1–d2 → BREVE
        if action_b == "NA":
            d1 = d2 = "NA"
        else:
            esiti_b = outcome_from_daily(df_daily, anchor_close, next_days[:2], action_b)
            d1, d2 = (esiti_b + ["NA", "NA"])[:2]

        # d3–d5 → MEDIO
        if action_m == "NA":
            d3 = d4 = d5 = "NA"
        else:
            esiti_m = outcome_from_daily(df_daily, anchor_close, next_days[2:5], action_m)
            d3, d4, d5 = (esiti_m + ["NA", "NA", "NA"])[:3]

        rows.append(BacktestRow(
            date=str(d0.date()),
            action_b=action_b, score_b=round(float(score_b), 3),
            action_m=action_m, score_m=round(float(score_m), 3),
            d1=d1, d2=d2, d3=d3, d4=d4, d5=d5
        ))

    df_out = pd.DataFrame([r.__dict__ for r in rows])
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Backtest split: BREVE (d1–d2) vs MEDIO (d3–d5)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Lista ticker es. AAPL MSFT GOOGL")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Quanti giorni indietro")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Soglia per LONG/SHORT (stessa per breve/medio)")
    parser.add_argument("--debug", action="store_true", help="Stampe di debug (anchor, giorni, H/L)")
    args = parser.parse_args()

    for t in args.tickers:
        tkr = t.upper()
        print(f"\n=== {tkr} | days_back={args.days_back} | threshold={args.threshold} ===")
        df = run_for_ticker(tkr, args.days_back, args.threshold, debug=args.debug)
        out_csv = f"backtest_split_{tkr}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[{tkr}] righe: {len(df)} → salvato {out_csv}")
        if not df.empty:
            print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
