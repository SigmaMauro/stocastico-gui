# -*- coding: utf-8 -*-
"""
Backtest BREVISSIMO con prezzi (anchor + estremi/close dei 5 giorni successivi)
-------------------------------------------------------------------------------
- Usa il check "brevissimo" per generare action/score (as-of il giorno t).
- Anchor = Close DAILY del giorno t.
- Per i 5 trading days successivi salva:
    * EXT (se LONG: High; se SHORT: Low)
    * CLOSE (Close daily)
- Weekend/festivi esclusi (si usano i trading days dal daily).

Esempio:
  python backtest_brevissimo_prezzi.py --tickers AAPL GOOGL --days-back 60 --threshold 0.30 --debug

Requisiti:
  pip install yfinance pandas numpy
  Il file 'stocastico_gui.py' deve stare nella stessa cartella (riusa stochastic/make_recommendations).
"""
import argparse
import datetime as dt
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Import dal tuo progetto (usa i tuoi indicatori e regole)
import stocastico_gui as sg


# ======================== DEFAULT EDITABILI ========================
DEFAULT_TICKERS: List[str] = ["googl", "msft","v","ma","mrna","pfe","spot","nflx"]
DEFAULT_DAYS_BACK: int = 30
DEFAULT_THRESHOLD: float = 0.30
# ==================================================================


# ---------------------- Preset scenari/indicatori (BREVISSIMO) ----
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
VOTE_VAL: Dict[str, int] = {"long": 1, "short": -1, "neutral": 0}


# ---------------------- Utility ----------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rende le colonne standard: Open/High/Low/Close/Volume anche se yfinance ritorna MultiIndex."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # Unisci livelli e poi mappa alla prima colonna che inizia con base OHLCV
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
                    pre_buffer_days: int = 240) -> pd.DataFrame:
    """Scarica abbastanza storico per calcolare indicatori **fino ad asof_dt** e restituisce df tagliato."""
    total_days = days + pre_buffer_days + 7
    period_all = yahoo_cap_period(total_days, interval)
    df = yf.download(ticker, period=period_all, interval=interval, auto_adjust=False, progress=False)
    df = normalize_ohlcv(df)
    if df is None or df.empty:
        raise RuntimeError("Dati non disponibili")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Mantieni solo barre fino a fine giornata as-of
    day_end = pd.Timestamp(asof_dt.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff_idx = df.index[df.index <= day_end]
    if len(cutoff_idx) == 0:
        # piccolo margine se l'ultima barra arriva dopo mezzanotte UTC
        fallback_limit = day_end + pd.Timedelta(hours=12)
        cutoff_idx = df.index[df.index <= fallback_limit]
        if len(cutoff_idx) == 0:
            raise RuntimeError("Nessuna barra entro la data As-Of.")
    last_asof_ts = cutoff_idx[-1]

    df_past = df.loc[:last_asof_ts]
    if df_past.shape[0] < 60:
        raise RuntimeError("Storico insufficiente prima della data As-Of.")
    return df_past


def scenario_score(df: pd.DataFrame, last_asof_ts: pd.Timestamp,
                   days: int, interval: str, ind_weights: Dict[str, float]) -> float:
    k_auto = recommended_k(days, interval)
    # Usa i tuoi indicatori
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
        df_past = fetch_data_asof(ticker, days, interval, asof_dt)
        last_ts = df_past.index[-1]
        s = scenario_score(df_past, last_ts, days, interval, ind_weights)
        total += s * scen_w
    return float(total)


# ---------------------- Backtest ----------------------
def run_for_ticker(ticker: str, days_back: int, threshold: float, debug: bool = False) -> pd.DataFrame:
    rows: List[dict] = []
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

    # trading days nel range richiesto
    trade_days = [d for d in df_daily.index if (start_date <= d.date() <= end_date)]
    print(f"[{ticker}] trading days da testare: {len(trade_days)} (da {start_date} a {end_date})")
    idx_list = list(df_daily.index)

    for d0 in trade_days:
        anchor_close = float(df_daily.at[d0, "Close"])
        asof_ts = pd.Timestamp(d0) + pd.Timedelta(hours=23, minutes=59)

        # Score/azione BREVISSIMO
        try:
            score_b = weighted_check_asof(ticker, asof_ts, BREVI_SCENARIOS, BREVI_IND_WEIGHTS)
        except Exception as e:
            if debug:
                print(f"  - {d0.date()} SKIP (errore calc breve: {e})")
            continue

        if score_b >= threshold:
            action = "LONG"; ext_label = "High"
        elif score_b <= -threshold:
            action = "SHORT"; ext_label = "Low"
        else:
            action = "NA"; ext_label = "—"

        # trova i 5 trading days successivi
        try:
            pos0 = idx_list.index(d0)
        except ValueError:
            if debug:
                print(f"  - {d0.date()} SKIP (giorno non in index)")
            continue
        next_days = [idx_list[pos0 + k] for k in (1, 2, 3, 4, 5) if (pos0 + k) < len(idx_list)]

        # prepara riga risultato
        row = {
            "date": str(d0.date()),
            "action": action,
            "score": round(float(score_b), 3),
            "anchor_close": round(anchor_close, 4),
            "ext_type": ext_label  # "High" se LONG, "Low" se SHORT, "—" se NA
        }

        # per ciascun giorno, salva EXT (High o Low in base all'azione) e CLOSE
        for i, dX in enumerate(next_days, start=1):
            hi = float(df_daily.at[dX, "High"])
            lo = float(df_daily.at[dX, "Low"])
            cl = float(df_daily.at[dX, "Close"])
            if action == "LONG":
                ext = hi
            elif action == "SHORT":
                ext = lo
            else:
                ext = np.nan
                cl = np.nan
            row[f"d{i}_ext"] = round(ext, 4) if not np.isnan(ext) else ""
            row[f"d{i}_close"] = round(cl, 4) if not np.isnan(cl) else ""

        # se mancano giorni futuri, riempi colonne mancanti
        for i in range(len(next_days) + 1, 6):
            row[f"d{i}_ext"] = ""
            row[f"d{i}_close"] = ""

        if debug:
            print(f"  {d0.date()} | action={action} score={row['score']} anchor={row['anchor_close']}")
            print(f"    next days: {[str(x.date()) for x in next_days]}")
            if action != "NA":
                print(f"    ext_type={ext_label}  d1..d5 ext/close:",
                      [(row[f'd{k}_ext'], row[f'd{k}_close']) for k in range(1,6)])

        rows.append(row)

    df_out = pd.DataFrame(rows, columns=[
        "date", "action", "score", "anchor_close", "ext_type",
        "d1_ext", "d1_close", "d2_ext", "d2_close", "d3_ext", "d3_close",
        "d4_ext", "d4_close", "d5_ext", "d5_close"
    ])
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Backtest BREVISSIMO con prezzi (ext + close per d1..d5)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Lista ticker es. AAPL MSFT GOOGL")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Quanti giorni indietro")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Soglia per LONG/SHORT")
    parser.add_argument("--debug", action="store_true", help="Stampe di debug (anchor, giorni, ext/close)")
    args = parser.parse_args()

    for t in args.tickers:
        tkr = t.upper()
        print(f"\n=== {tkr} | days_back={args.days_back} | threshold={args.threshold} ===")
        df = run_for_ticker(tkr, args.days_back, args.threshold, debug=args.debug)
        out_csv = f"backtest_breve_prezzi_{tkr}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[{tkr}] righe: {len(df)} → salvato {out_csv}")
        if not df.empty:
            print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
