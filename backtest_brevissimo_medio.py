
# -*- coding: utf-8 -*-
"""
Backtest "check brevissimo" e "check medio"
-------------------------------------------------
Cosa fa:
  - Per ogni giorno t nel range [oggi - DAYS_BACK, oggi - 7 giorni]
    calcola il punteggio "brevissimo" (1–2g) e "medio" (3–15g) **as-of** t,
    decide LONG/SHORT/NA in base a una soglia assoluta,
    e verifica se il segnale avrebbe "preso" nei giorni successivi.

Esito atteso per ciascun giorno testato: 5 colonne (giorni 1-5 dopo t):
  - 'SI'  se il target direzionale è stato toccato almeno una volta nel giorno (intraday High/Low)
  - 'NO'  se NON è stato toccato in quel giorno
  - 'NA'  se quel giorno non viene valutato (es: punteggio sotto soglia)

Regola di esito (semplice e robusta):
  - LONG  → successo se, nel giorno considerato, il massimo (High) supera il prezzo di chiusura alla data t
  - SHORT → successo se, nel giorno considerato, il minimo (Low) scende sotto la chiusura alla data t

Come usarlo (esempio):
  python backtest_brevissimo_medio.py --tickers AAPL MSFT TSLA --days-back 60 --threshold 0.30

Requisiti:
  pip install yfinance pandas numpy
  Questo script importa funzioni dal tuo file 'stocastico_gui.py' (stessa cartella).
"""
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

# Importa le funzioni dal tuo file esistente
# (il file ha il main protetto da if __name__ == '__main__', quindi l'import è sicuro)
import stocastico_gui as sg

# ---------------------- Config degli scenari (come nella tua GUI) ---------------------- #
# === CONFIG LOCALE ===
DEFAULT_TICKERS = ["googl", "msft","v","ma","mrna","pfe","spot","nflx"]
DEFAULT_DAYS_BACK = 20
DEFAULT_THRESHOLD = 0.30
# Brevissimo (1–2g) — timeframe + peso scenario e pesi indicatori
BREVI_SCENARIOS: List[Tuple[int,str,float]] = [
    (2,  "5m",  0.10),
    (5,  "15m", 0.20),
    (10, "30m", 0.40),
    (10, "1h",  0.25),
    (20, "1h",  0.05),
]
BREVI_IND_WEIGHTS: Dict[str,float] = {
    "Stoch": 0.40,
    "MACD": 0.20,
    "MACD_hist": 0.10,
    "RSI": 0.15,
    "Trend": 0.10,
    "OBV": 0.05,
}

# Medio (3–15g)
MEDIO_SCENARIOS: List[Tuple[int,str,float]] = [
    (3,  "30m", 0.20),
    (5,  "1h",  0.25),
    (10, "90m", 0.30),
    (15, "4h",  0.15),
    (30, "4h",  0.10),
]
MEDIO_IND_WEIGHTS: Dict[str,float] = {
    "Stoch": 0.15,
    "RSI": 0.15,
    "MACD": 0.25,
    "MACD_hist": 0.15,
    "OBV": 0.15,
    "Trend": 0.15,
}
VOTE_VAL = {"long": 1, "short": -1, "neutral": 0}

# ---------------------- Utility ---------------------- #
def bars_per_day(interval: str) -> float:
    mapping = {
        "1d": 1.0, "1h": 7.0, "90m": 4.5, "60m": 7.0, "30m": 13.0,
        "15m": 26.0, "5m": 78.0, "1m": 390.0, "4h": 1.8
    }
    return mapping.get(interval, 1.0)
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rende le colonne standard: Open/High/Low/Close/Volume anche se yfinance ritorna MultiIndex."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # unisci i livelli e poi prendi la prima occorrenza per ogni base OHLCV
        df = df.copy()
        df.columns = ['_'.join([str(x) for x in col if str(x) != '']) for col in df.columns.values]
        for base in ["Open", "High", "Low", "Close", "Volume"]:
            matches = [c for c in df.columns if c.startswith(base)]
            if matches:
                df[base] = df[matches[0]]
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        if keep:
            df = df[keep]
    return df
#yf.download
def recommended_k(days: int, interval: str, cap_bars: int | None = None) -> int:
    bars = int(round(days * bars_per_day(interval)))
    if cap_bars is not None:
        bars = min(bars, cap_bars)
    k = max(5, min(300, int(round(bars * 0.45))))
    k = max(5, min(k, max(5, bars - 2)))
    return k

def yahoo_cap_period(days: int, interval: str) -> str:
    if interval in ("1m",):
        return f"{min(days, 7)}d"
    elif interval in ("2m","5m","15m","30m","60m","90m","1h","2h","4h","6h"):
        return f"{min(days, 60)}d"
    else:
        return f"{days}d"

def fetch_data_asof(ticker: str, days: int, interval: str, asof_dt: pd.Timestamp,
                    pre_buffer_days: int = 240) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Scarica abbastanza storico per calcolare indicatori fino ad asof_dt e restituisce
    (df_past_fino_asof, last_asof_ts).
    """
    total_days = days + pre_buffer_days + 7  # 7g extra di margine
    period_all = yahoo_cap_period(total_days, interval)
    df_all = yf.download(ticker, period=period_all, interval=interval, auto_adjust=False, progress=False)
    df_all = normalize_ohlcv(df_all)  # <— aggiungi qui

    if df_all is None or df_all.empty:
        raise RuntimeError("Dati non disponibili")

    if isinstance(df_all.columns, pd.MultiIndex):
        df_all.columns = ['_'.join(col).strip() for col in df_all.columns.values]
        colmap = {}
        for base in ["Open","High","Low","Close","Volume"]:
            matches = [c for c in df_all.columns if c.startswith(base)]
            if matches: colmap[base] = matches[0]
        df_all = df_all.rename(columns={v:k for k,v in colmap.items() if v in df_all.columns})

    if getattr(df_all.index, "tz", None) is not None:
        df_all.index = df_all.index.tz_localize(None)
    df_all = df_all.sort_index()

    # ultima barra entro la fine della giornata as-of (23:59)
    day_end = pd.Timestamp(asof_dt.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff_idx = df_all.index[df_all.index <= day_end]
    if len(cutoff_idx) == 0:
        # fallback: entro mezzogiorno del giorno dopo
        fallback_limit = day_end + pd.Timedelta(hours=12)
        cutoff_idx = df_all.index[df_all.index <= fallback_limit]
        if len(cutoff_idx) == 0:
            raise RuntimeError("Nessuna barra entro la data As-Of per questo intervallo.")
    last_asof_ts = cutoff_idx[-1]

    df_past = df_all.loc[:last_asof_ts].tail(days + pre_buffer_days)
    if df_past.shape[0] < 60:
        raise RuntimeError("Storico insufficiente prima della data As-Of.")
    return df_past, last_asof_ts

def scenario_score(df: pd.DataFrame, last_asof_ts: pd.Timestamp,
                   days: int, interval: str, ind_weights: Dict[str,float]) -> float:
    k_auto = recommended_k(days, interval)
    # smoothing come nella GUI (3/3)
    _, sk, sd = sg.stochastic(df, k_period=k_auto, smooth_k=3, smooth_d=3)
    sk = sk.loc[:last_asof_ts]; sd = sd.loc[:last_asof_ts]
    rec = sg.make_recommendations(df.loc[:last_asof_ts], sk, sd)
    sig = rec.get("signals", {}) or {}
    scen = 0.0
    for ind, w in ind_weights.items():
        scen += VOTE_VAL.get(sig.get(ind, "neutral"), 0) * w
    return scen

def weighted_check_asof(ticker: str, asof_dt: pd.Timestamp,
                        scenarios: List[Tuple[int,str,float]],
                        ind_weights: Dict[str,float]) -> float:
    total = 0.0
    for days, interval, scen_w in scenarios:
        df_past, last_ts = fetch_data_asof(ticker, days, interval, asof_dt)
        s = scenario_score(df_past, last_ts, days, interval, ind_weights)
        total += s * scen_w
    return float(total)

def outcome_by_day(df_future: pd.DataFrame, anchor_close: float,
                   target_day: pd.Timestamp, side: str) -> str:
    day0 = pd.Timestamp(target_day.date())
    day1 = day0 + pd.Timedelta(days=1)
    mask = (df_future.index >= day0) & (df_future.index < day1)
    if not mask.any():
        return "NO"
    d = df_future.loc[mask]
    if d.empty:
        return "NO"

    # estrai scalari robusti
    if "High" not in d.columns or "Low" not in d.columns:
        return "NO"

    # High può essere Series o DataFrame → trasformo sempre in numpy e prendo il max/min scalare
    hi = float(np.nanmax(np.asarray(d["High"])))
    lo = float(np.nanmin(np.asarray(d["Low"])))

    if side == "LONG":
        return "SI" if hi > anchor_close else "NO"
    elif side == "SHORT":
        return "SI" if lo < anchor_close else "NO"
    return "NA"

@dataclass
class BacktestRow:
    date: str
    action: str
    score: float
    d1: str; d2: str; d3: str; d4: str; d5: str

def run_for_ticker(ticker: str, days_back: int, threshold: float, debug: bool = False) -> pd.DataFrame:
    rows: List[BacktestRow] = []
    today = dt.date.today()
    start_date = today - dt.timedelta(days=days_back)
    end_date = today - dt.timedelta(days=7)  # fino a una settimana fa

    # DAILY per anchor + outcome (robusto contro weekend/fusi)
    day_period = f"{days_back + 30}d"
    df_daily = yf.download(ticker, period=day_period, interval="1d", auto_adjust=False, progress=False)
    df_daily = normalize_ohlcv(df_daily)
    if df_daily is None or df_daily.empty:
        raise RuntimeError(f"Nessun daily per {ticker}")
    if getattr(df_daily.index, "tz", None) is not None:
        df_daily.index = df_daily.index.tz_localize(None)
    df_daily = df_daily.sort_index()

    # trading days nell'intervallo richiesto (solo dove c'è barra daily)
    trade_days = [d for d in df_daily.index if (start_date <= d.date() <= end_date)]
    print(f"[{ticker}] trading days da testare: {len(trade_days)} (da {start_date} a {end_date})")

    # Prepara lista indice per saltare weekend/festivi correttamente
    idx_list = list(df_daily.index)

    for d0 in trade_days:
        # 1) anchor = Close daily del giorno del segnale
        try:
            anchor_close = float(df_daily.at[d0, "Close"])
        except KeyError:
            # se per qualche motivo manca la colonna standardizzata, salta
            if debug:
                print(f"  - {d0.date()} SKIP (Close non disponibile)")
            continue

        # 2) calcolo punteggio "brevissimo" AS-OF quel giorno (entro 23:59)
        asof_ts = pd.Timestamp(d0) + pd.Timedelta(hours=23, minutes=59)
        try:
            score_b = weighted_check_asof(ticker, asof_ts, BREVI_SCENARIOS, BREVI_IND_WEIGHTS)
        except Exception as e:
            if debug:
                print(f"  - {d0.date()} SKIP (errore calc punteggio: {e})")
            continue

        # 3) determina direzione con soglia
        if score_b >= threshold:
            side = "LONG"
        elif score_b <= -threshold:
            side = "SHORT"
        else:
            side = "NA"

        # 4) giorni di trading successivi (non calendario)
        try:
            pos0 = idx_list.index(d0)
        except ValueError:
            if debug:
                print(f"  - {d0.date()} SKIP (giorno non in index)")
            continue
        next_days = [idx_list[pos0 + k] for k in (1, 2, 3, 4, 5) if (pos0 + k) < len(idx_list)]

        # --- DEBUG: stampa contesto del controllo
        if debug:
            highs_lows = []
            for x in next_days:
                try:
                    hi = float(df_daily.at[x, "High"])
                    lo = float(df_daily.at[x, "Low"])
                except KeyError:
                    hi, lo = float("nan"), float("nan")
                highs_lows.append((hi, lo))
            print(f"  {d0.date()} | action={side} score={score_b:.3f} anchor={anchor_close:.2f}")
            print(f"    next days: {[str(x.date()) for x in next_days]}")
            print(f"    H/L: {highs_lows}")

        # 5) esiti d1..d5 (sempre rispetto all'anchor del giorno d0)
        if side == "NA":
            esiti = ["NA"] * 5
        else:
            base = outcome_from_daily(df_daily, anchor_close, next_days, side)
            esiti = base + ["NA"] * (5 - len(base))

        rows.append(BacktestRow(
            date=str(d0.date()), action=side, score=round(float(score_b), 3),
            d1=esiti[0], d2=esiti[1], d3=esiti[2], d4=esiti[3], d5=esiti[4]
        ))

    df_out = pd.DataFrame([r.__dict__ for r in rows])
    return df_out
def outcome_from_daily(daily_df: pd.DataFrame,
                       anchor_close: float,
                       days_after: list[pd.Timestamp],
                       side: str) -> list[str]:
    # usa la tua outcome_by_day per ogni giorno successivo
    return [ outcome_by_day(daily_df, anchor_close, d, side) for d in days_after ]

def main():
    parser = argparse.ArgumentParser(description="Backtest check brevissimo / medio")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Lista ticker es. AAPL TSLA MSFT")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK,
                        help="Quanti giorni indietro (fino a 60 per intraday)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Soglia assoluta del punteggio per LONG/SHORT")
    args = parser.parse_args()

    all_results = {}
    for t in args.tickers:
        df = run_for_ticker(t.upper(), args.days_back, args.threshold)
        all_results[t.upper()] = df
        out_csv = f"backtest_{t.upper()}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[{t.upper()}] righe: {len(df)}  → salvato {out_csv}")
        if not df.empty:
            # mostrina di esempio
            print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
