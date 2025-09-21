import tkinter as tk
from tkinter import ttk, messagebox
from datetime import timezone

import mplfinance as mpf
import threading

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# -------------------------- Indicatori -------------------------- #

import datetime as dt

import re
import feedparser,  urllib.parse

import os, re, datetime as dt, threading, time
from typing import List, Dict

# sopprimi warning symlink su Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_FINBERT_PIPE = None
_FINBERT_LOCK = threading.Lock()
from transformers import pipeline

# cache il modello per non ricaricarlo ogni volta
_finbert_pipeline = None
def get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        import torch
        torch.set_num_threads(1)  # evita thread multipli
        _finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    return _finbert_pipeline

def analyze_with_finbert(news_items):
    """
    news_items: lista [(title, date, url), ...]
    Restituisce [(title, date, score, label), ...], overall_score
    """
    p = get_finbert()
    results = []
    score_sum = 0
    count = 0

    for (title, date, url) in news_items:
        out = p(title, truncation=True)[0]  # es: {'label': 'positive', 'score': 0.81}
        label = out['label'].lower()
        score = out['score'] if label == 'positive' else (-out['score'] if label == 'negative' else 0)
        results.append((title, date, url, score, label))
        score_sum += score
        count += 1

    overall = score_sum / count if count > 0 else 0
    return results, overall
def kelly_fraction(p: float, b: float) -> float:
    """
    Restituisce la frazione di capitale f* = p - (1-p)/b (Kelly classico).
    p = probabilità di successo (0..1)
    b = payoff ratio (media_win / media_loss), es. 2.0 significa vinco 2 quando perdo 1
    Clamp a [0, 1] e gestisce edge case.
    """
    if b is None or b <= 0 or p is None or p <= 0 or p >= 1:
        return 0.0
    f = p - (1.0 - p) / b
    return max(0.0, min(1.0, float(f)))


def kelly_half(f: float) -> float:
    "Mezzo-Kelly: più stabile in pratica."
    return max(0.0, min(1.0, float(f) * 0.5))

def load_finbert_blocking():
    """Carica FinBERT (blocking). Va eseguito SOLO in un thread separato."""
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    try:
        # opzionale: riduci verbosità transformers
        try:
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        model_name = "ProsusAI/finbert"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        # device=-1 => CPU (evita problemi GPU)
        _FINBERT_PIPE = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, device=-1)
        return _FINBERT_PIPE
    except Exception as e:
        _FINBERT_PIPE = None
        raise e

_COMP_RE = re.compile(r"\b(NVDA|NVIDIA|INTC|Intel|AMD|META|TSLA|AAPL|MSFT|GOOGL|AMZN|TSM|V|MA|PYPL|IBM|ORCL)\b", re.I)

def analyze_news_finbert_detailed(news_items: List[Dict], ticker: str):
    """
    Usa FinBERT. CHIAMARE SOLO DA THREAD NON-GUI.
    Ritorna: overall (-1..+1), tag, details[{title,publisher,time,link,score,sentiment}]
    """
    if not news_items:
        return 0.0, "none", []
    pipe = load_finbert_blocking()
    tk = (ticker or "").upper()

    # LIMITA il numero di news
    news_items = news_items[:10]

    titles = [(n.get("title") or "").strip() for n in news_items]
    outputs = pipe(titles, batch_size=4)

    details = []; total = 0.0
    for n, scores in zip(news_items, outputs):
        d = {s['label'].lower(): float(s['score']) for s in scores}
        p_pos = d.get("positive", 0.0); p_neg = d.get("negative", 0.0)
        base = p_pos - p_neg  # ∈ [-1,+1]

        title = (n.get("title") or "")
        m = _COMP_RE.search(title)
        if m and m.group(0).upper() != tk:
            if p_pos > p_neg: base -= 0.2
            elif p_neg > p_pos: base += 0.2

        score = max(-1.0, min(1.0, base))
        sent = "pos" if score > 0.15 else ("neg" if score < -0.15 else "neu")

        ts = n.get("time")
        if isinstance(ts, (int, float)):
            try: ts = dt.datetime.utcfromtimestamp(int(ts))
            except Exception: ts = None

        details.append({
            "title": (n.get("title") or "").strip(),
            "publisher": n.get("publisher",""),
            "time": ts,
            "link": n.get("link",""),
            "score": round(score, 3),
            "sentiment": sent
        })
        total += score

    avg = total / max(1, len(details))
    tag = "bullish" if avg > 0.25 else ("bearish" if avg < -0.25 else "mixed")
    return float(avg), tag, details
def run_news_analysis_async(use_finbert: bool, news: List[Dict], ticker: str, on_done, on_error, timeout_s=8.0):
    """
    Esegue analisi notizie in un thread e richiama on_done(overall, tag, items).
    Se FinBERT è disattivato o fallisce o scade il timeout → usa fallback analitico a regole.
    on_done/on_error DEVONO essere funzioni sicure per essere chiamate da thread (noi useremo .after nel caller).
    """
    def worker():
        start = time.time()
        overall = 0.0; n_tag = "none"; items = []
        try:
            if use_finbert:
                overall, n_tag, items = analyze_news_finbert_detailed(news or [], ticker)
            else:
                raise RuntimeError("FinBERT disattivato")
        except Exception:
            # fallback a regole
            try:
                ov, tg, it = analyze_news_bias_detailed(news or [], ticker)
                overall, n_tag, items = ov, tg, it
            except Exception as e2:
                on_error(e2); return
        # timeout soft: se ha impiegato troppo, segnala ma restituisci comunque
        if time.time() - start > timeout_s:
            # non è un errore, ma potresti voler loggare
            pass
        on_done(overall, n_tag, items)
    threading.Thread(target=worker, daemon=True).start()

def get_recent_news_rss(ticker: str, company: str | None = None, days: int = 5, max_items: int = 12):
    """
    Fallback RSS: Yahoo Finance + Google News.
    Ritorna [{'title','publisher','time','link'}].
    """
    out = []
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)

    feeds = [
        # Yahoo Finance RSS per ticker
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}",
        # Google News (tutte le fonti, query su ticker e, se noto, sul nome società)
        f"https://news.google.com/rss/search?q={urllib.parse.quote(ticker)}&hl=en-US&gl=US&ceid=US:en",
    ]
    if company:
        feeds.append(f"https://news.google.com/rss/search?q={urllib.parse.quote(company)}&hl=en-US&gl=US&ceid=US:en")

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = e.get("title", "").strip()
                link  = e.get("link", "")
                pub   = getattr(e, "source", {}).get("title", "") or e.get("publisher", "") or e.get("author", "") or ""
                # parse data
                ts = None
                for key in ("published_parsed","updated_parsed"):
                    val = e.get(key)
                    if val:
                        ts = dt.datetime(*val[:6])
                        break
                if not ts:  # salta se non sappiamo quando è uscita
                    continue
                if ts < cutoff:
                    continue
                out.append({"title": title, "publisher": pub, "time": ts, "link": link})
                if len(out) >= max_items:
                    return out
        except Exception:
            continue
    return out


def get_recent_news_yf(ticker: str, days: int = 5, max_items: int = 10):
    """
    Ritorna una lista di dict: [{'title', 'publisher', 'time', 'link'}] delle ultime notizie
    entro 'days' giorni. Usa yfinance Ticker.news (se disponibile).
    """
    out = []
    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None)
        if not news:
            return out
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
        for n in news[:max_items*2]:  # prendine un po' di più e poi filtra
            t_unix = n.get("providerPublishTime") or n.get("providerpublishTime") or n.get("published_at")
            if t_unix is None:
                continue
            try:
                ts = dt.datetime.utcfromtimestamp(int(t_unix))
            except Exception:
                continue
            if ts < cutoff:
                continue
            out.append({
                "title": n.get("title", "").strip(),
                "publisher": n.get("publisher") or n.get("provider") or "",
                "time": ts,
                "link": n.get("link") or n.get("url") or "",
            })
            if len(out) >= max_items:
                break
    except Exception:
        pass
    return out

_POS = r"(beats|surge|soars|jumps|rallies|spikes|upgrade|raises|boosts|strong|record|partnership|invests|acquires)"
_NEG = r"(misses|falls|sinks|tumbles|plunges|slumps|downgrade|cuts|warns|guidance\s+lower|probe|lawsuit|ban|halts)"
_comp_re = re.compile(r"\b(NVDA|NVIDIA|INTC|Intel|AMD|META|TSLA|AAPL|MSFT|GOOGL|AMZN|TSM)\b", re.I)

def analyze_news_bias(news_items: list[dict], ticker: str):
    """
    Ritorna (score, tag, bullets)
      - score in [-2..+2] (circa)
      - tag: 'bullish' | 'bearish' | 'mixed' | 'none'
      - bullets: lista di stringhe sintetiche per GUI
    Heuristica: +1 per headline positiva, -1 per negativa, +/-0.5 se cita competitor con verbi forti.
    """
    if not news_items:
        return 0.0, "none", []

    pos_re = re.compile(_POS, re.I)
    neg_re = re.compile(_NEG, re.I)
    score = 0.0
    bullets = []

    tk = ticker.upper()
    for n in news_items:
        title = n["title"]
        t = title.lower()

        # segno base sul titolo
        s = 0.0
        if pos_re.search(t): s += 1.0
        if neg_re.search(t): s -= 1.0

        # mention competitor: se positivo su competitor diretto → -0.5; se negativo su competitor → +0.5
        comp = _comp_re.search(title)
        if comp and comp.group(0).upper() != tk:
            # se headline ha parole positive e parla del competitor → penalizza (potrebbe spostare flussi)
            if pos_re.search(t): s -= 0.5
            if neg_re.search(t): s += 0.5

        score += s

        # bullet sintetico
        when = n["time"].strftime("%Y-%m-%d")
        src = n["publisher"] or ""
        bullets.append(f"[{when}] {src}: {title}")

    # normalizza in range ~[-2, +2]
    score = max(-2.0, min(2.0, score / max(1, len(news_items)) * 1.2))

    if score > 0.4:   tag = "bullish"
    elif score < -0.4: tag = "bearish"
    else:             tag = "mixed"

    return score, tag, bullets


# cache semplice in memoria
_EARNINGS_CACHE = {}

def get_earnings_dates_yf(ticker: str, limit: int = 16) -> list[dt.datetime]:
    """
    Restituisce una lista ordinata di datetime (UTC naive) delle date earnings note a Yahoo.
    Prova prima get_earnings_dates, poi calendar come fallback.
    Risultati cache-izzati nella sessione.
    """
    t = ticker.upper().strip()
    if t in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[t]

    dates: list[dt.datetime] = []
    try:
        tk = yf.Ticker(t)
        # 1) API principale
        try:
            df = tk.get_earnings_dates(limit=limit)
            if df is not None and not df.empty:
                col = "Earnings Date"
                if col in df.columns:
                    d = pd.to_datetime(df[col]).tz_localize(None)
                    dates.extend(d.to_pydatetime().tolist())
                else:
                    d = pd.to_datetime(df.index).tz_localize(None)
                    dates.extend(d.to_pydatetime().tolist())
        except Exception:
            pass
        # 2) Fallback: calendar
        if not dates:
            try:
                cal = tk.calendar
                # vari formati possibili, proviamo a pescare stringhe/ts
                if cal is not None and not cal.empty:
                    if "Earnings Date" in cal.index:
                        vals = cal.loc["Earnings Date"].values
                        for v in vals:
                            try:
                                d = pd.to_datetime(v).tz_localize(None).to_pydatetime()
                                dates.append(d)
                            except Exception:
                                continue
            except Exception:
                pass
    except Exception:
        pass

    dates = sorted({d for d in dates if isinstance(d, dt.datetime)})
    _EARNINGS_CACHE[t] = dates
    return dates


def earnings_proximity(ticker: str, ref_dt: dt.datetime, window_days: int = 5):
    """
    Per una data di riferimento (ref_dt, naive), ritorna:
    (prev_earning, next_earning, days_since_prev, days_until_next, flag_near)
    dove flag_near=True se next_earning è entro window_days.
    """
    ref = ref_dt
    dates = get_earnings_dates_yf(ticker)
    if not dates:
        return None, None, None, None, False

    prevs = [d for d in dates if d <= ref]
    nexts = [d for d in dates if d >= ref]

    prev_e = max(prevs) if prevs else None
    next_e = min(nexts) if nexts else None

    ds_prev = (ref - prev_e).days if prev_e else None
    du_next = (next_e - ref).days if next_e else None

    near = (du_next is not None) and (0 <= du_next <= window_days)
    return prev_e, next_e, ds_prev, du_next, near

def stochastic(df, k_period=14, smooth_k=3, smooth_d=3):
    low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    fast_k = 100 * (df['Close'] - low_min) / rng
    slow_k = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    slow_d = slow_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return fast_k, slow_k, slow_d

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
def adx(df, n=14):
    """
    Ritorna (adx, +DI, -DI) calcolati senza chained assignment.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movements (senza assegnazioni in-place)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_n = tr.rolling(n, min_periods=n).sum()
    plus_di = 100 * (plus_dm.rolling(n, min_periods=n).sum() / tr_n.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(n, min_periods=n).sum() / tr_n.replace(0, np.nan))

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(n, min_periods=n).mean()

    return adx, plus_di, minus_di

def obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index, name="OBV")

def sma(series, n):
    return series.rolling(n, min_periods=n).mean()

def atr(df, n=14):
    # True Range
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        (df['High'] - df['Low']).abs(),
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def last_crossover(slow_k, slow_d, lookback=10):
    """
    Ritorna ('bull'|'bear'|None, idx) se negli ultimi 'lookback' periodi
    c'è stato un incrocio %K/%D (da sotto->sopra = bull, da sopra->sotto = bear)
    """
    spread = slow_k - slow_d
    sig = np.sign(spread)
    # cerchiamo cambi di segno negli ultimi 'lookback' punti
    recent = sig.dropna().iloc[-(lookback+1):]
    if len(recent) < 2:
        return None, None
    # trova indice dell'ultimo cambio
    change_idx = None
    for i in range(1, len(recent)):
        if recent.iloc[i] == 0:
            continue
        if recent.iloc[i] != recent.iloc[i-1]:
            change_idx = recent.index[i]
    if change_idx is None:
        return None, None
    # classifica
    if spread.loc[change_idx] > 0:
        return 'bull', change_idx
    else:
        return 'bear', change_idx

def analyze_news_bias_detailed(news_items: list[dict], ticker: str):
        """
        Analizza le notizie e restituisce:
          overall_score, overall_tag, details

        details è una lista di dict con:
          {'title','publisher','time','link','score','sentiment'}  sentiment ∈ {'pos','neg','neu'}
        Regole (heuristiche):
          +1 per headline positiva, -1 per negativa;
          +/-0.5 se è positiva/negativa ma riferita a competitor diretto (penalizza o avvantaggia).
        """
        if not news_items:
            return 0.0, "none", []

        pos_re = re.compile(_POS, re.I)
        neg_re = re.compile(_NEG, re.I)
        tk = ticker.upper()
        details = []
        tot = 0.0

        for n in news_items:
            title = n.get("title", "").strip()
            t = title.lower()
            base = 0.0

            # segno base
            if pos_re.search(t): base += 1.0
            if neg_re.search(t): base -= 1.0

            # competitor effect
            comp_m = _comp_re.search(title)
            if comp_m:
                comp = comp_m.group(0).upper()
                if comp != tk:
                    # positivo sul competitor → penalizza il nostro
                    if pos_re.search(t): base -= 0.5
                    # negativo sul competitor → leggero beneficio per noi
                    if neg_re.search(t): base += 0.5

            # clamp leggero per evitare outlier
            score = max(-2.0, min(2.0, base))

            sentiment = "neu"
            if score > 0.15:
                sentiment = "pos"
            elif score < -0.15:
                sentiment = "neg"

            details.append({
                "title": title,
                "publisher": n.get("publisher", ""),
                "time": n.get("time"),
                "link": n.get("link", ""),
                "score": score,
                "sentiment": sentiment
            })

            tot += score

        # normalizza media in ~[-2,+2] come prima
        avg = tot / max(1, len(details))
        overall = max(-2.0, min(2.0, avg * 1.2))
        if overall > 0.4:
            tag = "bullish"
        elif overall < -0.4:
            tag = "bearish"
        else:
            tag = "mixed"

        return overall, tag, details

# -------------------------- Regole consigli -------------------------- #
def make_recommendations(df, slow_k, slow_d, atr_period=14):
    """
    Ritorna:
    {
      'short_term': {'action','confidence','reason','leverage'},
      'long_term':  {'action','confidence','reason','leverage'},
      'risk':       {'level','value_pct'}
    }
    """

    # ---------- Helper locali (RSI, MACD, confidenza, leve) ----------
    def _rsi_tag(value: float) -> str:
        """
        Restituisce 'long' | 'short' | 'neutral' secondo le fasce:
          <25  -> long (ipervenduto)
          25-45-> short (momentum debole)
          45-55-> neutral
          55-75-> long  (momentum)
          >75  -> short (ipercomprato)
        """
        if not pd.notna(value):
            return 'neutral'
        if value < 25:
            return 'long'
        if value < 45:
            return 'short'
        if value <= 55:
            return 'neutral'
        if value <= 75:
            return 'long'
        return 'short'
    def rsi_local(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(n, min_periods=n).mean()
        avg_loss = loss.rolling(n, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def macd_local(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def _bump_conf(conf: str, up=True):
        order = ['bassa', 'media', 'alta']
        i = order.index(conf) if conf in order else 0
        i = min(2, i + 1) if up else max(0, i - 1)
        return order[i]

    def _map_conf_to_leverage(horizon: str, confidence: str, action: str, atr_pct: float | None):
        """
        horizon: 'short' (1–7g) oppure 'long' (swing).
        - Short: leva ATR-dominante continua in range 3–10
        - Long:  rimappata automaticamente in 1.5–4.5
        Confidenza (RSI/MACD) ora pesa di più:
          alta +3.0 | media +1.5 | bassa -0.5
        """
        # se non c'è segnale o ATR, minima leva per l'orizzonte
        if action == "ASTIENITI" or atr_pct is None or np.isnan(atr_pct):
            return 3 if horizon == "short" else 1.5

        # --- ATR dominante, continuo: clamp 0.5%..6%, 0.5% -> 10 ; 6% -> 3
        atr_c = max(0.5, min(6.0, float(atr_pct)))
        lev_atr = 10.636 + (-1.2727) * atr_c  # retta per (0.5,10) e (6,3)

        # --- Influenza confidenza (più forte)
        if confidence == "alta":
            bonus = 3.0
        elif confidence == "media":
            bonus = 0
        else:  # "bassa" o altro
            bonus = -3 # penalizza quando RSI/MACD non confermano

        lev_short = lev_atr + bonus
        lev_short = max(3.0, min(10.0, lev_short))  # clamp 3–10

        if horizon == "short":
            return int(round(lev_short))

        # --- Rimappatura per il LONG: 3..10 -> 1.5..4.5
        lev_long = 1.5 + (lev_short - 3.0) * (3.0 / 7.0)
        lev_long = max(1.5, min(4.5, lev_long))
        return round(lev_long, 1)
    # ---------- Guardie su dati ----------
    if df is None or df.empty or slow_k is None or slow_d is None:
        base = {
            'short_term': {'action': 'ASTIENITI', 'confidence': 'bassa', 'reason': 'Dati non disponibili', 'leverage': 3},
            'long_term':  {'action': 'ASTIENITI', 'confidence': 'bassa', 'reason': 'Dati non disponibili', 'leverage': 1.5},
            'risk':       {'level': 'N/D', 'value_pct': None}
        }
        return base

    # Allinea e pulisci le serie stocastiche
    sk = slow_k.dropna()
    sd = slow_d.dropna()
    common = sk.index.intersection(sd.index)
    sk = sk.loc[common]
    sd = sd.loc[common]
    if len(sk) < 5 or len(sd) < 5:
        base = {
            'short_term': {'action': 'ASTIENITI', 'confidence': 'bassa', 'reason': 'Storico insufficiente', 'leverage': 3},
            'long_term':  {'action': 'ASTIENITI', 'confidence': 'bassa', 'reason': 'Storico insufficiente', 'leverage': 1.5},
            'risk':       {'level': 'N/D', 'value_pct': None}
        }
        return base

    # ---------- Rischio (ATR%) ----------
    # usa un ritaglio sufficiente per stabilità; atr_period è configurabile dalla GUI
    df_for_atr = df.tail(max(atr_period * 5, 100))
    atrn = atr(df_for_atr, atr_period)
    last_atr = atrn.iloc[-1] if atrn.notna().any() else np.nan
    last_close = df_for_atr['Close'].iloc[-1]
    atr_pct = float(last_atr / last_close * 100) if pd.notna(last_atr) else None

    if atr_pct is None:
        risk_level = 'N/D'
    elif atr_pct > 3:
        risk_level = 'ALTO'
    elif atr_pct > 1:
        risk_level = 'MEDIO'
    else:
        risk_level = 'BASSO'
    risk = {'level': risk_level, 'value_pct': round(atr_pct, 2) if atr_pct is not None else None}

    # ---------- Short term (stocastico + conferme RSI/MACD) ----------
    k = sk.iloc[-1]; d = sd.iloc[-1]
    k_prev = sk.iloc[-2] if len(sk) >= 2 else np.nan
    d_prev = sd.iloc[-2] if len(sd) >= 2 else np.nan
    slope_k = k - k_prev

    cross, _ = last_crossover(sk, sd, lookback=10)
    st_action, st_conf, st_reason = 'ASTIENITI', 'bassa', []

    if cross == 'bull':
        st_action = 'LONG'; st_reason.append("incrocio rialzista %K>%D recente")
        if k < 20: st_conf = 'alta'; st_reason.append("in ipervenduto (<20)")
        elif k > 50 and slope_k > 0: st_conf = 'media'; st_reason.append("%K>50 e in aumento")
        else: st_conf = 'media'
    elif cross == 'bear':
        st_action = 'SHORT'; st_reason.append("incrocio ribassista %K<%D recente")
        if k > 80: st_conf = 'alta'; st_reason.append("in ipercomprato (>80)")
        elif k < 50 and slope_k < 0: st_conf = 'media'; st_reason.append("%K<50 e in calo")
        else: st_conf = 'media'
    else:
        if k > d and k > 50 and slope_k > 0:
            st_action, st_conf = 'LONG', 'bassa'; st_reason.append("%K>%D, >50 e in aumento")
        elif k < d and k < 50 and slope_k < 0:
            st_action, st_conf = 'SHORT', 'bassa'; st_reason.append("%K<%D, <50 e in calo")
        else:
            st_reason.append("segnali deboli/contrastanti")

    # RSI/MACD conferme (breve)
    rsi14 = rsi_local(df['Close'], 14)
    macd_line, macd_signal, macd_hist = macd_local(df['Close'])
    rsi_last = rsi14.iloc[-1] if rsi14.notna().any() else np.nan
    macd_last = macd_line.iloc[-1] if macd_line.notna().any() else np.nan
    signal_last = macd_signal.iloc[-1] if macd_signal.notna().any() else np.nan
    hist_last = macd_hist.iloc[-1] if macd_hist.notna().any() else np.nan

    # ---- Conferme RSI/MACD per il BREVE (RSI doppia logica) ----
    rsi_tag = _rsi_tag(rsi_last)
    confirm_msgs = []
    # ---- ADX & OBV ----
    adx_val, plus_di, minus_di = adx(df, n=14)
    adx_last = adx_val.iloc[-1] if not adx_val.empty else np.nan
    plus_last = plus_di.iloc[-1] if not plus_di.empty else np.nan
    minus_last = minus_di.iloc[-1] if not minus_di.empty else np.nan

    obv_val = obv(df)
    obv_last = obv_val.iloc[-1] if not obv_val.empty else np.nan
    obv_prev = obv_val.iloc[-2] if len(obv_val) > 1 else np.nan

    # effetto RSI sul segnale attuale
    if st_action == 'LONG':
        if rsi_tag == 'long':
            st_conf = _bump_conf(st_conf, up=True);
            confirm_msgs.append("RSI favorevole al LONG")
            # doppio boost se estremo (ipervenduto <25)
            if pd.notna(rsi_last) and rsi_last < 25:
                st_conf = _bump_conf(st_conf, up=True);
                confirm_msgs.append("RSI<25 (ipervenduto): spinta extra")
        elif rsi_tag == 'short':
            st_conf = _bump_conf(st_conf, up=False);
            confirm_msgs.append("RSI sfavorevole (verso SHORT)")
            # doppia penalità se >75
            if pd.notna(rsi_last) and rsi_last > 75:
                st_conf = _bump_conf(st_conf, up=False);
                confirm_msgs.append("RSI>75 (ipercomprato): freno forte")
    elif st_action == 'SHORT':
        if rsi_tag == 'short':
            st_conf = _bump_conf(st_conf, up=True);
            confirm_msgs.append("RSI favorevole allo SHORT")
            if pd.notna(rsi_last) and rsi_last > 75:
                st_conf = _bump_conf(st_conf, up=True);
                confirm_msgs.append("RSI>75 (ipercomprato): spinta extra")
        elif rsi_tag == 'long':
            st_conf = _bump_conf(st_conf, up=False);
            confirm_msgs.append("RSI sfavorevole (verso LONG)")
            if pd.notna(rsi_last) and rsi_last < 25:
                st_conf = _bump_conf(st_conf, up=False);
                confirm_msgs.append("RSI<25 (ipervenduto): freno forte")

    # MACD come prima (direzione + istogramma)
    if pd.notna(macd_last) and pd.notna(signal_last):
        if st_action == 'LONG' and macd_last > signal_last:
            st_conf = _bump_conf(st_conf, up=True);
            confirm_msgs.append("MACD sopra Signal")
        elif st_action == 'SHORT' and macd_last < signal_last:
            st_conf = _bump_conf(st_conf, up=True);
            confirm_msgs.append("MACD sotto Signal")
        elif st_action in ('LONG', 'SHORT'):
            st_conf = _bump_conf(st_conf, up=False);
            confirm_msgs.append("MACD contrario")

    if pd.notna(hist_last):
        if st_action == 'LONG' and hist_last > 0:
            confirm_msgs.append("Istogramma MACD > 0")
        elif st_action == 'SHORT' and hist_last < 0:
            confirm_msgs.append("Istogramma MACD < 0")
        # ADX: se trend forte (>25) alza un po' la confidenza
        if pd.notna(adx_last) and adx_last > 25:
            if st_conf == 'bassa':
                st_conf = _bump_conf(st_conf, up=True)
            confirm_msgs.append(f"ADX {round(adx_last, 1)}: trend forte")

        # OBV: se va nella stessa direzione, rafforza
        if pd.notna(obv_last) and pd.notna(obv_prev):
            if st_action == 'LONG' and obv_last > obv_prev:
                st_conf = _bump_conf(st_conf, up=True)
                confirm_msgs.append("OBV crescente conferma LONG")
            elif st_action == 'SHORT' and obv_last < obv_prev:
                st_conf = _bump_conf(st_conf, up=True)
                confirm_msgs.append("OBV decrescente conferma SHORT")


    if confirm_msgs:
        st_reason.append(" ; ".join(confirm_msgs))



    short_term = {
        'action': st_action,
        'confidence': st_conf,
        'reason': " | ".join(st_reason) if st_reason else "—",
    }

    # Penalizza confidenza se rischio ALTO (ma non toccare se già 'alta')
    if risk['level'] == 'ALTO' and short_term['action'] != 'ASTIENITI' and short_term['confidence'] != 'alta':
        short_term['confidence'] = 'bassa'
        short_term['reason'] += " | volatilità alta: prudenza"

    # ---------- Long term (trend SMA200 + conferme RSI/MACD) ----------
    sma200_series = sma(df['Close'], 200)
    c = df['Close'].iloc[-1]
    s200 = sma200_series.iloc[-1] if pd.notna(sma200_series.iloc[-1]) else np.nan

    long_action, long_conf, reasons_lt = 'ASTIENITI', 'bassa', []
    if not np.isnan(s200):
        if c > s200 * 1.01:
            if k < 90:
                long_action = 'LONG'; reasons_lt.append("prezzo sopra SMA200 (trend up)")
                if k > d: long_conf = 'media'; reasons_lt.append("%K>%D")
                else:     long_conf = 'bassa'
            else:
                reasons_lt.append("ipercomprato estremo (>90): attesa")
        elif c < s200 * 0.99:
            if k > 10:
                long_action = 'SHORT'; reasons_lt.append("prezzo sotto SMA200 (trend down)")
                if k < d: long_conf = 'media'; reasons_lt.append("%K<%D")
                else:     long_conf = 'bassa'
            else:
                reasons_lt.append("ipervenduto estremo (<10): rischio rimbalzo")
        else:
            reasons_lt.append("vicino a SMA200: indecisione")
    else:
        reasons_lt.append("SMA200 non disponibile")

    lt = {
        'action': long_action,
        'confidence': long_conf,
        'reason': " | ".join(reasons_lt) if reasons_lt else "—"
    }

    # Conferme soft per il lungo con RSI a doppia logica
    rsi_tag = _rsi_tag(rsi_last)
    if lt['action'] == 'LONG':
        if rsi_tag == 'long':
            lt['confidence'] = _bump_conf(lt['confidence'], up=True)
            lt['reason'] += " | RSI favorevole al LONG"
        elif rsi_tag == 'short':
            lt['confidence'] = _bump_conf(lt['confidence'], up=False)
            lt['reason'] += " | RSI sfavorevole (verso SHORT)"
    elif lt['action'] == 'SHORT':
        if rsi_tag == 'short':
            lt['confidence'] = _bump_conf(lt['confidence'], up=True)
            lt['reason'] += " | RSI favorevole allo SHORT"
        elif rsi_tag == 'long':
            lt['confidence'] = _bump_conf(lt['confidence'], up=False)
            lt['reason'] += " | RSI sfavorevole (verso LONG)"

    # ---------- Leve suggerite ----------
    short_term['leverage'] = _map_conf_to_leverage("short", short_term['confidence'], short_term['action'], risk['value_pct'])
    lt['leverage']         = _map_conf_to_leverage("long",  lt['confidence'],         lt['action'],         risk['value_pct'])

    # --------- Semafori segnale (per la GUI) ---------
    def _tag(val: str):  # helper per rendere esplicito
        return val  # 'long' | 'short' | 'neutral'

    signals = {}

    # RSI: >55 long, <45 short, altrimenti neutro
    signals['RSI'] = rsi_tag

    # MACD vs Signal
    if pd.notna(macd_last) and pd.notna(signal_last):
        signals['MACD'] = _tag(
            'long' if macd_last > signal_last else ('short' if macd_last < signal_last else 'neutral'))
    else:
        signals['MACD'] = _tag('neutral')

    # Istogramma MACD (sopra/sotto zero)
    if pd.notna(hist_last):
        signals['MACD_hist'] = _tag('long' if hist_last > 0 else ('short' if hist_last < 0 else 'neutral'))
    else:
        signals['MACD_hist'] = _tag('neutral')

    # Stocastico: direzione del suggerimento breve
    signals['Stoch'] = _tag(
        'long' if short_term['action'] == 'LONG' else ('short' if short_term['action'] == 'SHORT' else 'neutral'))

    # Trend di fondo: prezzo vs SMA200 (con buffer)
    if not np.isnan(s200):
        if c > s200 * 1.01:
            signals['Trend'] = _tag('long')
        elif c < s200 * 0.99:
            signals['Trend'] = _tag('short')
        else:
            signals['Trend'] = _tag('neutral')
    else:
        signals['Trend'] = _tag('neutral')

    # ATR: non direzionale -> sempre neutro (giallo)
    signals['ATR'] = _tag('neutral')
    # ADX: verde/rosso solo se >25, altrimenti neutro
    if pd.notna(adx_last) and adx_last > 25:
        signals['ADX'] = 'long' if plus_last > minus_last else 'short'
    else:
        signals['ADX'] = 'neutral'

    # OBV: confronta ultima variazione
    if pd.notna(obv_last) and pd.notna(obv_prev):
        if obv_last > obv_prev:
            signals['OBV'] = 'long'
        elif obv_last < obv_prev:
            signals['OBV'] = 'short'
        else:
            signals['OBV'] = 'neutral'
    else:
        signals['OBV'] = 'neutral'
    # ---------- Kelly: stima p (da confidenza) e b (da ATR → R:R) ----------
    def _p_from_conf(conf: str) -> float:
        # stima conservativa: bassa ~52%, media ~55%, alta ~58%
        return {"bassa": 0.52, "media": 0.55, "alta": 0.58}.get(conf, 0.50)

    def _b_from_risk_level(risk_level: str) -> float:
        # Se la volatilità è alta, mantieni R:R più prudente (1.2),
        # altrimenti 1.5 (medio) o 2.0 (basso) come target reward:stop.
        if risk_level == "ALTO":  return 1.2
        if risk_level == "MEDIO": return 1.5
        if risk_level == "BASSO": return 2.0
        return 1.5

    b_rr = _b_from_risk_level(risk.get("level", "N/D"))

    # --- Kelly per breve ---
    if short_term['action'] in ("LONG", "SHORT"):
        p_st = _p_from_conf(short_term['confidence'])
        f_st = kelly_fraction(p_st, b_rr)
        short_term['kelly_fraction'] = round(f_st, 4)
        short_term['kelly_half']     = round(kelly_half(f_st), 4)
        short_term.setdefault('reason', '')
        short_term['reason'] += f" | Kelly½~{short_term['kelly_half']*100:.1f}% (p~{int(p_st*100)}%, R:R~{b_rr:.1f})"
    else:
        short_term['kelly_fraction'] = 0.0
        short_term['kelly_half'] = 0.0

    # --- Kelly per lungo ---
    if lt['action'] in ("LONG", "SHORT"):
        p_lt = _p_from_conf(lt['confidence'])
        f_lt = kelly_fraction(p_lt, b_rr)
        lt['kelly_fraction'] = round(f_lt, 4)
        lt['kelly_half']     = round(kelly_half(f_lt), 4)
        lt.setdefault('reason', '')
        lt['reason'] += f" | Kelly½~{lt['kelly_half']*100:.1f}% (p~{int(p_lt*100)}%, R:R~{b_rr:.1f})"
    else:
        lt['kelly_fraction'] = 0.0
        lt['kelly_half'] = 0.0

    return {'short_term': short_term, 'long_term': lt, 'risk': risk, 'signals': signals}


# -------------------------- GUI App -------------------------- #
class StochasticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self._sig_colors = {
            'long': {'fg': 'white', 'bg': '#2e7d32'},  # verde
            'short': {'fg': 'white', 'bg': '#c62828'},  # rosso
            'neutral': {'fg': 'black', 'bg': '#f9a825'},  # giallo
        }

        self.title("Grafico Azione + Stocastico")
        # nella __init__
        self.geometry("1200x820")  # un po' più larga



        self._build_controls()
        # --- Splitter verticale: top (grafico) / bottom (consigli+status) ---
        self.pw = tk.PanedWindow(self, orient=tk.VERTICAL)
        self.pw.pack(fill=tk.BOTH, expand=True)

        self.top_area = ttk.Frame(self.pw)  # per il grafico
        self.bottom_area = ttk.Frame(self.pw)  # per consigli + status
        self.pw.add(self.top_area)  # più spazio al grafico
        self.pw.add(self.bottom_area, minsize=180)  # area minima visibile
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # costruisci contenuti nelle due aree
        self._build_plot(parent=self.top_area)
        self._build_reco_panel(parent=self.bottom_area)
        self._build_statusbar(parent=self.bottom_area)
        self.bind("<Configure>", self._on_resize)
        self.last_df = None
        self.last_ticker = None
        self.last_rec = None

    def _on_close(self):
        import matplotlib.pyplot as plt
        plt.close('all')  # chiude tutte le figure matplotlib
        self.destroy()  # chiude finestra principale
        self.quit()

    def _compose_judgment_text(self, rec: dict, ref_dt: dt.datetime | None = None) -> str:
        import datetime as _dt
        try:
            signals = rec.get('signals', {}) or {}
            risk = rec.get('risk', {}) or {}
            st = rec.get('short_term', {}) or {}
            lt = rec.get('long_term', {}) or {}

            # inferisci orizzonte (riusa il tuo metodo se già c'è)
            horizon = self._infer_horizon() if hasattr(self, "_infer_horizon") else "breve"

            vote_val = {'long': 1, 'short': -1, 'neutral': 0}
            keys = ["Stoch", "RSI", "MACD", "MACD_hist", "Trend", "OBV"]
            score = sum(vote_val.get(signals.get(k, 'neutral'), 0) for k in keys)

            adx_state = signals.get("ADX", "neutral")
            adx_val = signals.get("ADX_value", None)
            trend_strong = adx_state in ("long", "short")

            atr_level = risk.get("level", "N/D")
            atr_pct = risk.get("value_pct", None)

            lines = []

            if horizon == "breve":
                base_dir = st.get("action", "ASTIENITI");
                base_conf = st.get("confidence", "bassa")
                lines.append(f"Breve periodo: **{base_dir}** (conf. {base_conf})." if base_dir != "ASTIENITI"
                             else "Breve periodo: **Astenersi** — segnale principale debole/contrasto.")
            elif horizon == "medio":
                base_dir = st.get("action", "ASTIENITI");
                trend = signals.get("Trend", "neutral")
                if base_dir != "ASTIENITI" and trend != "neutral" and (
                        (base_dir == "LONG" and trend == "long") or (base_dir == "SHORT" and trend == "short")):
                    lines.append(f"Medio periodo: **{base_dir}** — stocastico allineato al trend (SMA200).")
                elif base_dir != "ASTIENITI" and trend != "neutral":
                    lines.append("Medio periodo: **prudenza** — stocastico contro trend (pullback).")
                else:
                    lines.append("Medio periodo: **neutro/prudenza** — segnali misti.")
            else:
                base_dir = lt.get("action", "ASTIENITI");
                base_conf = lt.get("confidence", "bassa")
                lines.append(f"Lungo periodo: **{base_dir}** (conf. {base_conf})." if base_dir != "ASTIENITI"
                             else "Lungo periodo: **Astenersi** — prezzo vicino a SMA200 o segnali deboli.")

            trio = [signals.get("Stoch", "neutral"), signals.get("RSI", "neutral"), signals.get("MACD", "neutral")]
            if trio.count("long") >= 2 and signals.get("OBV", "neutral") == "long":
                lines.append("Combinazione: **Stoch/RSI/MACD pro-LONG + OBV in aumento** ⇒ segnale **forte LONG**.")
            if trio.count("short") >= 2 and signals.get("OBV", "neutral") == "short":
                lines.append("Combinazione: **Stoch/RSI/MACD pro-SHORT + OBV in calo** ⇒ segnale **forte SHORT**.")

            if signals.get("Stoch") == "short" and not trend_strong:
                lines.append("Nota: **Stoch SHORT con ADX basso** ⇒ trend debole/laterale → prudenza.")
            if signals.get("Trend") == "long" and signals.get("Stoch") == "short":
                lines.append("Divergenza: **Trend LONG ma Stoch SHORT** ⇒ probabile pullback.")
            if signals.get("Trend") == "short" and signals.get("Stoch") == "long":
                lines.append("Divergenza: **Trend SHORT ma Stoch LONG** ⇒ possibile rimbalzo tecnico.")

            if atr_level == "ALTO":
                extra = f" (~{atr_pct}%)" if atr_pct is not None else ""
                lines.append(f"Rischio: **ATR ALTO{extra}** ⇒ ridurre la leva, affidabilità minore.")
            elif atr_level == "BASSO":
                extra = f" (~{atr_pct}%)" if atr_pct is not None else ""
                lines.append(f"Rischio: **ATR BASSO{extra}** ⇒ contesto più stabile (ma occhio ai falsi segnali).")

            if trend_strong:
                dir_txt = "rialzista" if adx_state == "long" else "ribassista"
                if adx_val is not None:
                    lines.append(f"Forza trend: **ADX {adx_val} (>25)** ⇒ trend {dir_txt} robusto.")
                else:
                    lines.append(f"Forza trend: **ADX >25** ⇒ trend {dir_txt} robusto.")

            if score >= 3:
                lines.append("Voto badge: **maggioranza forte LONG**.")
            elif score <= -3:
                lines.append("Voto badge: **maggioranza forte SHORT**.")
            elif abs(score) == 2:
                lines.append("Voto badge: **maggioranza debole**.")
            else:
                lines.append("Voto badge: **disaccordo/laterale**.")

            if ref_dt is None:
                ref_dt = _dt.datetime.utcnow()

            tkr = self.last_ticker or self._parse_ticker(self.ticker_var.get())
            if tkr:
                prev_e, next_e, ds_prev, du_next, near = earnings_proximity(tkr, ref_dt, window_days=5)
                if near and next_e:
                    lines.append(f"⚠️ Earnings tra {du_next}g ({next_e.strftime('%Y-%m-%d')}): "
                                 f"indicatori tecnici meno affidabili → ridurre leva/valutare astensione.")
                elif ds_prev is not None and ds_prev <= 2 and prev_e:
                    lines.append(f"⚠️ Earnings appena pubblicati ({prev_e.strftime('%Y-%m-%d')}): "
                                 f"volatilità elevata → segnali instabili per breve periodo.")

            return "\n".join(lines)
        except Exception as e:
            return f"Giudizio non disponibile: {e}"

    def _bars_per_day(self, interval: str) -> float:
        # stime pratiche per mercati USA; vanno bene come default
        mapping = {
            "1d": 1.0,
            "1h": 7.0,  # ~6.5h, arrotondo a 7
            "90m": 4.5,
            "60m": 7.0,
            "30m": 13.0,
            "15m": 26.0,
            "5m": 78.0,
            "1m": 390.0,
            "1wk": 0.2,  # ~1 barra ogni 5 giorni
        }
        return mapping.get(interval, 1.0)

    def _recommended_k(self, days: int, interval: str, cap_bars: int | None = None) -> int:
        bars = int(round(days * self._bars_per_day(interval)))
        if cap_bars is not None:
            bars = min(bars, cap_bars)
        # target: ~45% della finestra
        k = max(5, min(300, int(round(bars * 0.45))))
        # evita K > numero barre disponibili
        k = max(5, min(k, max(5, bars - 2)))
        return k

    def _maybe_update_k(self):
        if not getattr(self, "auto_k_var", None) or not self.auto_k_var.get():
            return
        try:
            days = int(self.days_var.get())
            if days <= 0:
                return
        except Exception:
            return
        interval = self.interval_var.get()
        # opzionale: stima cap sulle barre massime davvero scaricabili (per intraday Yahoo)
        cap = None
        if interval in ("1m",):
            cap = min(days, 7) * int(self._bars_per_day(interval))
        elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h"):
            cap = min(days, 60) * int(self._bars_per_day(interval))
        k = self._recommended_k(days, interval, cap_bars=cap)
        self.k_period_var.set(str(k))

    def _build_statusbar(self, parent=None):
        parent = parent or self
        self.status_frame = ttk.Frame(parent, padding=(10, 4, 10, 4))
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ultimo aggiornamento: —")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

    def _format_last_time_text(self, ts):
        """
        ts: pandas.Timestamp (può essere timezone-aware o naive).
        Mostra orario della barra nell'orario dell'exchange (quello interno a ts, se presente)
        e anche nel tuo orario locale (sistema).
        """
        # pandas.Timestamp -> datetime
        py_dt = ts.to_pydatetime()
        # Se è naive, assumiamo UTC (yfinance a volte restituisce UTC su alcuni set)
        if py_dt.tzinfo is None:
            py_dt = py_dt.replace(tzinfo=timezone.utc)

        # Orario locale macchina
        local_dt = py_dt.astimezone()  # usa timezone locale del sistema

        exch_str = py_dt.strftime("%Y-%m-%d %H:%M %Z")
        local_str = local_dt.strftime("%Y-%m-%d %H:%M %Z")

        return f"Ultimo aggiornamento barra: {exch_str}  (locale: {local_str})"

    def _apply_preset(self, event=None):
        preset = self.preset_var.get()
        if preset == "Breve":
            self.days_var.set("20")
            self.interval_var.set("1h")
        elif preset == "Medio":
            self.days_var.set("90")
            self.interval_var.set("1d")
        elif preset == "Lungo":
            self.days_var.set("365")
            self.interval_var.set("1d")
        # se torna a "Personalizzato", non tocchiamo nulla



    def _brevissimo_check_weighted(self):
        """
        Analizza 5 scenari con pesi su timeframe e indicatori.
        Ritorna giudizio complessivo sul brevissimo (1-2 giorni).
        """
        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            messagebox.showerror("Errore", "Inserisci un ticker valido")
            return

        # --- Scenari e pesi timeframe
        scenarios = [
            ("2g, 5m", 2, "5m", 0.10),
            ("5g, 15m", 5, "15m", 0.20),
            ("10g, 30m", 10, "30m", 0.40),
            ("10g, 1h", 10, "1h", 0.25),
            ("20g, 1h", 20, "1h", 0.05),
        ]

        # --- Pesi indicatori
        weights_ind = {
            "Stoch": 0.40,
            "MACD": 0.20,
            "MACD_hist": 0.10,
            "RSI": 0.15,
            "Trend": 0.10,
            "OBV": 0.05,
        }

        top = tk.Toplevel(self)
        top.title(f"Brevissimo weighted check — {ticker}")
        top.geometry("950x600")

        text = tk.Text(top, wrap="word", font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True)

        total_score = 0.0
        details = []

        for label, days, interval, scen_weight in scenarios:
            try:
                df = yf.download(tickers=ticker, period=f"{days}d", interval=interval,
                                 auto_adjust=False, progress=False)
                if df is None or df.empty:
                    raise ValueError("Nessun dato")

                # Normalizza colonne
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    colmap = {}
                    for base in ["Open", "High", "Low", "Close", "Volume"]:
                        matches = [c for c in df.columns if c.startswith(base)]
                        if matches:
                            colmap[base] = matches[0]
                    df = df.rename(columns={v: k for k, v in colmap.items() if v in df.columns})

                fast_k, slow_k, slow_d = stochastic(df, k_period=14, smooth_k=3, smooth_d=3)
                rec = make_recommendations(df, slow_k, slow_d)

                signals = rec.get("signals", {})
                vote_val = {"long": 1, "short": -1, "neutral": 0}

                # punteggio scenario (somma pesata indicatori)
                scen_score = 0.0
                for ind, w in weights_ind.items():
                    sig = signals.get(ind, "neutral")
                    scen_score += vote_val.get(sig, 0) * w

                scen_score *= scen_weight
                total_score += scen_score

                details.append((label, scen_score, signals))

                # Dettagli scenario
                text.insert(tk.END, f"=== {label} ===\n")
                text.insert(tk.END, f"Segnali: {signals}\n")
                text.insert(tk.END, f"Punteggio scenario (pesato): {scen_score:.2f}\n\n")

            except Exception as e:
                text.insert(tk.END, f"Errore scenario {label}: {e}\n\n")

        # --- Sintesi finale ---
        text.insert(tk.END, "\n=== Sintesi complessiva ===\n")
        if total_score > 0.25:
            text.insert(tk.END, f"👉 Score {total_score:.2f} → Probabile rialzo (LONG breve termine)\n")
        elif total_score < -0.25:
            text.insert(tk.END, f"👉 Score {total_score:.2f} → Probabile ribasso (SHORT breve termine)\n")
        else:
            text.insert(tk.END, f"👉 Score {total_score:.2f} → Segnali misti/laterali, prudenza\n")
        # === IA contestuale: Notizie recenti (ultimi 5 giorni) ===
        try:
            news = get_recent_news_yf(ticker, days=5, max_items=8)
            if not news:
                # prova anche con il nome azienda se lo hai (es. INTC -> Intel)
                company_name = {

                    "NFLX": "Netflix",
                    "TSLA": "Tesla",
                    "MA": "Mastercard",
                    "META": "Meta",
                    "SPOT": "Spotify",
                    "GOOGL": "Alphabet (A)",
                    "MSFT": "Microsoft",
                    "NVDA": "NVIDIA",
                    "AMD": "AMD",
                    "V": "Visa",
                    "AMZN": "Amazon",
                    "AAPL": "Apple",
                    "MCD": "McDonald's",
                    "STM": "STMicroelectronics",
                    "SBUX": "Starbucks",
                    "BRBY.L": "Burberry Group",
                    "INTC": "Intel",
                    "MRNA": "Moderna",
                    "PFE": "Pfizer",
                    # aggiungi altri mapping che usi spesso
                }.get(ticker.upper(), None)
                news = get_recent_news_rss(ticker, company=company_name, days=5, max_items=8)

            if not news:
                text.insert(tk.END, "Nessuna notizia recente rilevata.\n")
            else:
                # crea tag colori una volta (subito prima di stampare le news)
                # crea i colori/tag per il testo
                text.tag_configure("pos", foreground="#128a00")
                text.tag_configure("neg", foreground="#c62828")
                text.tag_configure("neu", foreground="#b8860b")

                text.insert(tk.END, "\n=== IA contestuale — Notizie recenti (5g) ===\n")

                # placeholder immediato (GUI NON SI BLOCCA)
                text.insert(tk.END, "Analisi IA in corso...\n")

                def _on_done(overall, n_tag, items):
                    def _update_gui():
                        # tag colore già creati prima: "pos", "neg", "neu"
                        if not items:
                            text.insert(tk.END, "Nessuna notizia recente rilevata.\n")
                        else:
                            for it in items[:5]:
                                title = it.get("title", "")
                                src = it.get("publisher", "")
                                ts = it.get("time", None)
                                when = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else ""
                                scr = it.get("score", 0.0)
                                sent = it.get("sentiment", "neu")
                                tag = "pos" if sent == "pos" else ("neg" if sent == "neg" else "neu")
                                line = f"• [{when}] {src}: {title}  ({scr:+.2f})\n"
                                text.insert(tk.END, line, tag)

                            text.insert(tk.END, "\n")
                            if n_tag == "bullish":
                                text.insert(tk.END, f"Conclusione IA: bias positivo (score {overall:+.2f}).\n", "pos")
                            elif n_tag == "bearish":
                                text.insert(tk.END, f"Conclusione IA: bias negativo (score {overall:+.2f}).\n", "neg")
                            else:
                                text.insert(tk.END, f"Conclusione IA: news miste/neutre (score {overall:+.2f}).\n",
                                           "neu")

                    text.after(0, _update_gui)

                def _on_error(e):
                    def _upd():
                        text.insert(tk.END, f"[IA notizie] Non disponibile: {e}\n")

                    text.after(0, _upd)

                use_finbert = bool(getattr(self, "use_finbert_var", None) and self.use_finbert_var.get())
                run_news_analysis_async(use_finbert, news, ticker, _on_done, _on_error, timeout_s=8.0)

        except Exception as e:
            text.insert(tk.END, f"\n[IA notizie] Non disponibile: {e}\n")

    def _medium_check_weighted(self):
        """
        Check medio termine (3–15 giorni).
        - Scenari: 3g/30m, 5g/1h, 10g/2h, 15g/4h
        - Pesi timeframe: 0.20, 0.30, 0.30, 0.20
        - Pesi indicatori (medio): Stoch .15, RSI .15, MACD .25, Hist .15, OBV .15, Trend .15
        - News IA: solo commento in fondo (non influenza il punteggio)
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        import yfinance as yf
        import pandas as pd

        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            messagebox.showerror("Errore", "Inserisci un ticker valido")
            return

        scenarios = [
            ("3g, 30m", 3, "30m", 0.20),
            ("5g, 1h", 5, "1h", 0.25),
            ("10g, 90m", 10, "90m", 0.30),
            ("15g, 4h", 15, "4h", 0.15),
            ("30g, 4h", 30, "4h", 0.10),
        ]

        # Pesi indicatori per medio termine
        weights_ind = {
            "Stoch": 0.15,
            "RSI": 0.15,
            "MACD": 0.25,
            "MACD_hist": 0.15,
            "OBV": 0.15,
            "Trend": 0.15,
        }
        vote_val = {"long": 1, "short": -1, "neutral": 0}

        top = tk.Toplevel(self)
        top.title(f"Medio termine (3–15g) — {ticker}")
        top.geometry("980x640")

        # area testo con scrollbar
        frm = ttk.Frame(top);
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt = tk.Text(frm, wrap="word", font=("Consolas", 10))
        vsb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        frm.grid_rowconfigure(0, weight=1);
        frm.grid_columnconfigure(0, weight=1)

        total_score = 0.0

        for label, days, interval, scen_w in scenarios:
            try:
                # cap period per intraday (coerente con il resto della tua app)
                if interval in ("1m",):
                    period = f"{min(days, 7)}d"
                elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h", "2h", "4h", "6h"):
                    period = f"{min(days, 60)}d"
                else:
                    period = f"{days}d"

                df = yf.download(tickers=ticker, period=period, interval=interval,
                                 auto_adjust=False, progress=False)
                if df is None or df.empty:
                    raise ValueError("Nessun dato")

                # normalizza colonne OHLC se MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    colmap = {}
                    for base in ["Open", "High", "Low", "Close", "Volume"]:
                        matches = [c for c in df.columns if c.startswith(base)]
                        if matches: colmap[base] = matches[0]
                    df = df.rename(columns={v: k for k, v in colmap.items() if v in df.columns})

                # parametri stocastico: usa auto-K già presente nella tua app
                k_auto = self._recommended_k(days, interval)
                smooth_k = int(self.smooth_k_var.get() or "3")
                smooth_d = int(self.smooth_d_var.get() or "3")
                _, slow_k, slow_d = stochastic(df, k_period=k_auto, smooth_k=smooth_k, smooth_d=smooth_d)

                rec = make_recommendations(df, slow_k, slow_d)  # già calcola signals

                signals = rec.get("signals", {}) or {}
                scen_score = 0.0
                for ind, w in weights_ind.items():
                    scen_score += vote_val.get(signals.get(ind, "neutral"), 0) * w

                scen_score *= scen_w
                total_score += scen_score

                # stampa blocco scenario
                txt.insert(tk.END, f"=== {label} ===\n")
                txt.insert(tk.END, f"Segnali: {signals}\n")
                # mostra anche ATR livello (utile per capire affidabilità)
                rk = rec.get("risk", {})
                atrline = f"ATR: {rk.get('level', 'N/D')}"
                if rk.get('value_pct') is not None:
                    atrline += f" ~ {rk['value_pct']}%"
                txt.insert(tk.END, atrline + "\n")
                txt.insert(tk.END, f"Punteggio scenario (pesato): {scen_score:.2f}\n\n")

            except Exception as e:
                txt.insert(tk.END, f"Errore scenario {label}: {e}\n\n")

        # conclusione
        txt.insert(tk.END, "=== Sintesi complessiva (medio termine) ===\n")
        if total_score > 0.40:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS LONG **moderato/forte** (3–15g)\n")
        elif total_score > 0.15:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS LONG **debole** (prudenza)\n")
        elif total_score < -0.40:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS SHORT **moderato/forte** (3–15g)\n")
        elif total_score < -0.15:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS SHORT **debole** (prudenza)\n")
        else:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → Segnali **misti/laterali**\n")

        # === IA contestuale: SOLO COMMENTO (non pesa nel punteggio) ===
        try:
            # prova yfinance.news; se vuoto, fallback RSS (se hai aggiunto gli helper)
            news = get_recent_news_yf(ticker, days=5, max_items=8)
            if not news:
                # usa mapping company_name se lo hai nel codice
                company_name = {
                    "NFLX": "Netflix", "TSLA": "Tesla", "MA": "Mastercard", "META": "Meta",
                    "SPOT": "Spotify", "GOOGL": "Alphabet (A)", "MSFT": "Microsoft", "NVDA": "NVIDIA",
                    "AMD": "AMD", "V": "Visa", "AMZN": "Amazon", "AAPL": "Apple", "MCD": "McDonald's",
                    "STM": "STMicroelectronics", "SBUX": "Starbucks", "BRBY.L": "Burberry Group",
                    "INTC": "Intel", "MRNA": "Moderna", "PFE": "Pfizer"
                }.get(ticker.upper(), None)
                try:
                    # richiede feedparser se usi il fallback RSS
                    news = get_recent_news_rss(ticker, company=company_name, days=5, max_items=8)
                except Exception:
                    news = []

            txt.insert(tk.END, "\n=== Nota IA (news recenti, SOLO COMMENTO) ===\n")
            if not news:
                txt.insert(tk.END, "Nessuna notizia rilevante negli ultimi 5 giorni.\n")
            else:
                # crea tag colori una volta (subito prima di stampare le news)
                # crea i colori/tag per il testo
                txt.tag_configure("pos", foreground="#128a00")
                txt.tag_configure("neg", foreground="#c62828")
                txt.tag_configure("neu", foreground="#b8860b")

                txt.insert(tk.END, "\n=== IA contestuale — Notizie recenti (5g) ===\n")

                # placeholder immediato (GUI NON SI BLOCCA)
                txt.insert(tk.END, "Analisi IA in corso...\n")

                def _on_done(overall, n_tag, items):
                    def _update_gui():
                        # tag colore già creati prima: "pos", "neg", "neu"
                        if not items:
                            txt.insert(tk.END, "Nessuna notizia recente rilevata.\n")
                        else:
                            for it in items[:5]:
                                title = it.get("title", "")
                                src = it.get("publisher", "")
                                ts = it.get("time", None)
                                when = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else ""
                                scr = it.get("score", 0.0)
                                sent = it.get("sentiment", "neu")
                                tag = "pos" if sent == "pos" else ("neg" if sent == "neg" else "neu")
                                line = f"• [{when}] {src}: {title}  ({scr:+.2f})\n"
                                txt.insert(tk.END, line, tag)

                            txt.insert(tk.END, "\n")
                            if n_tag == "bullish":
                                txt.insert(tk.END, f"Conclusione IA: bias positivo (score {overall:+.2f}).\n", "pos")
                            elif n_tag == "bearish":
                                txt.insert(tk.END, f"Conclusione IA: bias negativo (score {overall:+.2f}).\n", "neg")
                            else:
                                txt.insert(tk.END, f"Conclusione IA: news miste/neutre (score {overall:+.2f}).\n",
                                           "neu")

                    txt.after(0, _update_gui)

                def _on_error(e):
                    def _upd():
                        txt.insert(tk.END, f"[IA notizie] Non disponibile: {e}\n")

                    txt.after(0, _upd)

                use_finbert = bool(getattr(self, "use_finbert_var", None) and self.use_finbert_var.get())
                run_news_analysis_async(use_finbert, news, ticker, _on_done, _on_error, timeout_s=8.0)

        except Exception as e:
            txt.insert(tk.END, f"\n[IA news] Non disponibile: {e}\n")

    def _long_check_weighted(self):
        """
        Check lungo (1–2 mesi).
        Scenari: 30g/1h, 60g/4h, 90g/1d
        Pesi timeframe: 0.30, 0.40, 0.30
        Pesi indicatori: Trend .35, MACD .25, Hist .10, RSI .10, Stoch .10, OBV .10
        ATR: solo nota di rischio (non entra nello score direzionale)
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        import yfinance as yf
        import pandas as pd

        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            messagebox.showerror("Errore", "Inserisci un ticker valido")
            return

        scenarios = [
            ("30g, 1h", 30, "1h", 0.20),
            ("60g, 4h", 60, "4h", 0.25),
            ("100g, 1d", 90, "1d", 0.25),
            ("360g, 1d", 360, "1d", 0.20),
            ("700g, 1wk", 700, "1wk", 0.10),  # NEW: lunghissimo, pesa forte
        ]

        # Pesi indicatori (lungo)
        weights_ind = {
            "Trend": 0.40,
            "MACD": 0.25,
            "MACD_hist": 0.10,
            "RSI": 0.10,
            "Stoch": 0.05,
            "OBV": 0.10,
        }
        vote_val = {"long": 1, "short": -1, "neutral": 0}

        # finestra output
        top = tk.Toplevel(self)
        top.title(f"Lungo (1–2 mesi) — {ticker}")
        top.geometry("980x640")
        frm = ttk.Frame(top);
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt = tk.Text(frm, wrap="word", font=("Consolas", 10))
        vsb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        frm.grid_rowconfigure(0, weight=1);
        frm.grid_columnconfigure(0, weight=1)

        total_score = 0.0

        for label, days, interval, scen_w in scenarios:
            try:
                # cap period coerente con Yahoo + boost storico per 1wk
                if interval == "1m":
                    period = f"{min(days, 7)}d"
                elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"):
                    period = f"{min(days, 60)}d"
                elif interval == "1wk":
                    # per avere SMA200 su weekly servono ~200 settimane (~1400g)
                    fetch_days = max(days, 1500)  # scarica più storico dietro le quinte
                    period = f"{fetch_days}d"
                else:
                    period = f"{days}d"

                df = yf.download(tickers=ticker, period=period, interval=interval,
                                 auto_adjust=False, progress=False)
                if df is None or df.empty:
                    raise ValueError("Nessun dato")

                # normalizza colonne OHLC se MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    colmap = {}
                    for base in ["Open", "High", "Low", "Close", "Volume"]:
                        matches = [c for c in df.columns if c.startswith(base)]
                        if matches: colmap[base] = matches[0]
                    df = df.rename(columns={v: k for k, v in colmap.items() if v in df.columns})

                # stocastico con K auto sull'orizzonte scelto
                k_auto = self._recommended_k(days, interval)
                smooth_k = int(self.smooth_k_var.get() or "3")
                smooth_d = int(self.smooth_d_var.get() or "3")
                _, slow_k, slow_d = stochastic(df, k_period=k_auto, smooth_k=smooth_k, smooth_d=smooth_d)

                rec = make_recommendations(df, slow_k, slow_d)  # include signals + risk

                signals = rec.get("signals", {}) or {}
                scen_score = 0.0
                for ind, w in weights_ind.items():
                    scen_score += vote_val.get(signals.get(ind, "neutral"), 0) * w

                scen_score *= scen_w
                total_score += scen_score

                # stampa blocco scenario
                txt.insert(tk.END, f"=== {label} ===\n")
                txt.insert(tk.END, f"Segnali: {signals}\n")
                rk = rec.get("risk", {})
                atrline = f"ATR: {rk.get('level', 'N/D')}"
                if rk.get('value_pct') is not None:
                    atrline += f" ~ {rk['value_pct']}%"
                txt.insert(tk.END, atrline + "\n")
                txt.insert(tk.END, f"Punteggio scenario (pesato): {scen_score:.2f}\n\n")

            except Exception as e:
                txt.insert(tk.END, f"Errore scenario {label}: {e}\n\n")

        # conclusione (soglie più “lente”)
        txt.insert(tk.END, "=== Sintesi complessiva (lungo) ===\n")
        if total_score > 0.40:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS LONG **moderato/forte** (1–2 mesi)\n")
        elif total_score > 0.15:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS LONG **debole** (prudenza)\n")
        elif total_score < -0.40:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS SHORT **moderato/forte** (1–2 mesi)\n")
        elif total_score < -0.15:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → BIAS SHORT **debole** (prudenza)\n")
        else:
            txt.insert(tk.END, f"👉 Score {total_score:.2f} → Segnali **misti/laterali**\n")


    def _build_controls(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(side=tk.TOP, fill=tk.X)
        # ATR period (scelta preimpostata)
        ttk.Label(frm, text="ATR period:").grid(row=0, column=5, sticky="w", padx=(246, 8))
        self.atr_period_var = tk.StringVar(value="14")
        atr_values = ["7", "14", "21", "28", "35", "42", "49", "56", "63", "70"]
        atr_cb = ttk.Combobox(frm, textvariable=self.atr_period_var, width=5, state="readonly", values=atr_values)
        atr_cb.grid(row=0, column=5, sticky="w", padx=(326, 8))

        ttk.Label(frm, text="Ticker (es. TSLA):").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ticker_var = tk.StringVar(value="TSLA")
        ttk.Entry(frm, textvariable=self.ticker_var, width=15).grid(row=0, column=1, sticky="e",padx=(0, 15))

        ttk.Label(frm, text="Giorni:").grid(row=0, column=2, sticky="w", padx=(8, 8))
        self.days_var = tk.StringVar(value="30")
        ttk.Entry(frm, textvariable=self.days_var, width=8).grid(row=0, column=2, sticky="e",padx=(0, 8))

        # Parametri stocastico
        ttk.Label(frm, text="K period:").grid(row=0, column=3, sticky="w", padx=(8, 8))
        self.k_period_var = tk.StringVar(value="14")
        k_entry = ttk.Entry(frm, textvariable=self.k_period_var, width=6)
        k_entry.grid(row=0, column=3, sticky="e", padx=60)

        # CheckBox per auto K
        self.auto_k_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Auto K (consigliato)", variable=self.auto_k_var, command=self._maybe_update_k).grid(
            row=0, column=4, sticky="w", padx=(12, 0))

        ttk.Label(frm, text="smooth K:").grid(row=0, column=5, sticky="w", padx=(16, 8))
        self.smooth_k_var = tk.StringVar(value="3")
        ttk.Entry(frm, textvariable=self.smooth_k_var, width=6).grid(row=0, column=5, sticky="w", padx=90)

        ttk.Label(frm, text="smooth D:").grid(row=0, column=5, sticky="w", padx=130)
        self.smooth_d_var = tk.StringVar(value="3")
        ttk.Entry(frm, textvariable=self.smooth_d_var, width=6).grid(row=0, column=5, sticky="w", padx=(206, 0))

        ttk.Label(frm, text="Intervallo:").grid(row=0, column=6, sticky="e", padx=(0, 8))
        self.interval_var = tk.StringVar(value="1d")
        interval_cb = ttk.Combobox(frm, textvariable=self.interval_var, width=7, state="readonly",
                                   values=["1m", "5m", "15m", "30m", "60m", "90m", "1h","4h", "1d", "1wk"])
        interval_cb.grid(row=0, column=7, sticky="w")
        # Preset di orizzonte
        ttk.Label(frm, text="Orizzonte:").grid(row=0, column=7, sticky="w", padx=(66, 8))
        self.preset_var = tk.StringVar(value="Personalizzato")
        preset_cb = ttk.Combobox(frm, textvariable=self.preset_var, width=15, state="readonly",
                                 values=["Personalizzato", "Breve", "Medio", "Lungo"])
        preset_cb.grid(row=0, column=7, sticky="w", padx=(156, 8))
        preset_cb.bind("<<ComboboxSelected>>", self._apply_preset)

        self.btn = ttk.Button(frm, text="Crea grafico", command=self.on_plot_clicked)
        self.btn.grid(row=2, column=5, padx=(0, 0))
        self.btn_candles = ttk.Button(frm, text="Apri candele", command=self._open_candles_window)
        self.btn_candles.grid(row=2, column=4, padx=(0, 0))
        self.btn_multitf = ttk.Button(frm, text="Multi-TF Check", command=self.on_multi_tf_clicked)
        self.btn_multitf.grid(row=2, column=3, padx=(0, 0))
        # Data as-of (YYYY-MM-DD) e lookahead
        ttk.Label(frm, text="As-Of (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.asof_var = tk.StringVar(value="")  # vuoto = oggi/tempo reale
        ttk.Entry(frm, textvariable=self.asof_var, width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="Verifica (giorni):").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=(6, 0))
        self.lookahead_var = tk.StringVar(value="7")
        ttk.Entry(frm, textvariable=self.lookahead_var, width=6).grid(row=1, column=3, sticky="w", pady=(6, 0))

        self.btn_asof = ttk.Button(frm, text="Analizza a quella data", command=self.on_asof_clicked)
        self.btn_asof.grid(row=1, column=4, padx=(16, 0), pady=(6, 0))

        # se usi grid_columnconfigure alla fine, aumenta il range:
        # Bottone As-Of Multi-TF
        self.btn_asof_multitf = ttk.Button(frm, text="As-Of Multi-TF", command=self.on_asof_multi_clicked)
        self.btn_asof_multitf.grid(row=1, column=5, padx=(12, 0), pady=(6, 0))

        self.btn_brevissimo_w = ttk.Button(frm, text="Check Brevissimo (pesato)",
                                           command=self._brevissimo_check_weighted)
        self.btn_brevissimo_w.grid(row=1, column=6, padx=(8, 0))
        self.use_news_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Notizie (No IA)", variable=self.use_news_var).grid(row=1, column=7, padx=(8, 0))
        self.use_finbert_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="IA Notizie (FinBERT)", variable=self.use_finbert_var).grid(row=2, column=3,
                                                                                           padx=(8, 0))

        self.btn_medium_w = ttk.Button(frm, text="Check Medio (3–15g)", command=self._medium_check_weighted)
        self.btn_medium_w.grid(row=2, column=6, padx=(8, 0))
        self.btn_long_w = ttk.Button(frm, text="Check Lungo (1–2 mesi)", command=self._long_check_weighted)
        self.btn_long_w.grid(row=2, column=7, padx=(8, 0))

        for i in range(20):
            frm.grid_columnconfigure(i, weight=0)
        ttk.Button(frm, text="Apri legenda", command=self.show_legenda).grid(
            row=2, column=0, sticky="w", pady=(8,0)
        )

    def on_asof_multi_clicked(self):
        self.btn_asof_multitf.config(state=tk.DISABLED)
        threading.Thread(target=self._asof_multi_safe, daemon=True).start()

    def _asof_multi_safe(self):
        try:
            self._asof_multi_check()
        except Exception as e:
            messagebox.showerror("Errore As-Of Multi-TF", str(e))
        finally:
            self.btn_asof_multitf.config(state=tk.NORMAL)

    def _make_scroll_table(self, parent, cols, headers, widths, height=18):
        """
        Crea una Treeview con scroll vertical+horizontal.
        Ritorna (frame, tree).
        """
        wrap = ttk.Frame(parent)
        wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Tree + barre
        tree = ttk.Treeview(wrap, columns=cols, show="headings", height=height)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Layout: tree sopra, hsb sotto, vsb a destra
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        # Colonne
        for c, w, h in zip(cols, widths, headers):
            tree.heading(c, text=h)
            # stretch=False per permettere lo scroll orizzontale
            tree.column(c, width=w, minwidth=40, stretch=False, anchor="w")

        # Scorrimento orizzontale con Shift+rotella (comodo su Windows)
        tree.bind_all("<Shift-MouseWheel>",
                      lambda e: tree.xview_scroll(-1 if e.delta > 0 else 1, "units"))

        return wrap, tree

    def _asof_multi_check(self):
        import datetime as dt
        import pandas as pd

        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            raise ValueError("Inserisci un ticker valido.")

        asof_str = (self.asof_var.get() or "").strip()
        if not asof_str:
            raise ValueError("Inserisci una data As-Of nel formato YYYY-MM-DD (es. 2025-06-07).")
        try:
            asof_dt = dt.datetime.strptime(asof_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Formato data non valido. Usa YYYY-MM-DD.")

        # Giorni di verifica
        try:
            lookahead = int(self.lookahead_var.get())
            if lookahead <= 0: raise ValueError
        except Exception:
            raise ValueError("Lookahead deve essere un intero positivo.")

        # Scenari richiesti
        scenarios = [
            ("5g / 5m", 5, "5m"),
            ("20g / 30m", 20, "30m"),
            ("20g / 1h", 20, "1h"),
            ("40g / 1h", 40, "1h"),
            ("90g / 1h", 90, "1h"),
            ("365g / 1d", 365, "1d"),
        ]

        # Finestra risultati
        top = tk.Toplevel(self)
        top.title(f"As-Of Multi-TF — {ticker} @ {asof_str}")
        top.geometry("1200x560")

        hdr = ttk.Frame(top);
        hdr.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(hdr, text=f"Ticker: {ticker}  |  As-Of: {asof_str}  |  Lookahead: {lookahead}g",
                  font=("", 11, "bold")).pack(side=tk.LEFT)
        pbar = ttk.Progressbar(hdr, mode="determinate", maximum=len(scenarios), length=300)
        pbar.pack(side=tk.RIGHT, padx=8)

        # Tabella: tutti i badge + ATR + NOTE (giudizio + voto + esito)
        cols = ("scenario", "stoch", "rsi", "macd", "hist", "trend", "adx", "obv", "atr", "note")
        widths = (140, 70, 70, 70, 70, 80, 60, 60, 100, 620)
        headers = ("SCENARIO", "STOCH", "RSI", "MACD", "MACD_H", "TREND", "ADX", "OBV", "ATR", "NOTE")

        tree = ttk.Treeview(top, columns=cols, show="headings", height=20)
        for c, w, h in zip(cols, widths, headers):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="w")
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.place(in_=tree, relx=1.0, rely=0, relheight=1.0, x=0, y=0, anchor="ne")

        # Helpers
        def _cap_period(days, interval):
            if interval in ("1m",):
                return f"{min(days, 7)}d"
            elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h", "6h"):
                return f"{min(days, 60)}d"
            else:
                return f"{days}d"

        def _fmt_badge(sig, k):
            v = sig.get(k, None)
            return "—" if not v else v.upper()

        def _badge_vote_text(sig):
            vote_val = {'long': 1, 'short': -1, 'neutral': 0}
            keys = ["Stoch", "RSI", "MACD", "MACD_hist", "Trend", "OBV"]
            score = sum(vote_val.get(sig.get(k, 'neutral'), 0) for k in keys)
            if score >= 3:   return "Maggioranza forte LONG"
            if score <= -3:  return "Maggioranza forte SHORT"
            if abs(score) == 2: return "Maggioranza debole"
            return "Disaccordo/laterale"

        # Loop scenari
        now_utc = dt.datetime.utcnow()
        intraday_limits = {
            "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
            "60m": 730, "90m": 730, "1h": 730
        }

        for i, (label, days, interval) in enumerate(scenarios, start=1):
            try:
                # ---- controlla il limite storico Yahoo per l'intervallo scelto
                age_days = (now_utc - asof_dt).days
                if interval in intraday_limits and age_days > intraday_limits[interval]:
                    tree.insert("", "end", values=(
                        label, "—", "—", "—", "—", "—", "—", "—",
                        "—",
                        f"Limite Yahoo: {interval} disponibile solo ~{intraday_limits[interval]} giorni da oggi"
                    ))
                    pbar['value'] = i;
                    top.update_idletasks()
                    continue

                # ---- finestra start/end centrata sull’As-Of
                pre_buffer_days = max(200, int(days * 0.5))
                start = asof_dt - dt.timedelta(days=pre_buffer_days + days)
                end = asof_dt + dt.timedelta(days=lookahead + 1)  # +1 per includere l'ultimo giorno

                # ---- limites storici Yahoo per intraday
                now_utc = dt.datetime.utcnow()
                intraday_limits = {
                    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
                    "60m": 730, "90m": 730, "1h": 730
                }

                # ... dentro il for ogni scenario:
                # finestra teorica centrata su As-Of
                pre_buffer_days = max(200, int(days * 0.5))
                start = asof_dt - dt.timedelta(days=pre_buffer_days + days)
                end = asof_dt + dt.timedelta(days=lookahead + 1)

                # clamp ai limiti Yahoo (rispetto ad oggi)
                if interval in intraday_limits:
                    limit_days = intraday_limits[interval]
                    hard_start = now_utc - dt.timedelta(days=limit_days)
                    # non chiedere prima di quanto Yahoo tenga in memoria
                    if start < hard_start:
                        start = hard_start
                # non andare nel "futuro"
                if end > now_utc + dt.timedelta(days=1):
                    end = now_utc + dt.timedelta(days=1)

                # 1° tentativo: start/end clampati
                df_all = yf.download(
                    tickers=ticker, start=start, end=end,
                    interval=interval, auto_adjust=False, progress=False
                )

                # Fallback: se vuoto, usa period massimo consentito e poi taglia
                if df_all is None or df_all.empty:
                    if interval in intraday_limits:
                        period_fallback = f"{intraday_limits[interval]}d"
                    else:
                        period_fallback = f"{max(days + pre_buffer_days + lookahead, 400)}d"
                    df_all = yf.download(
                        tickers=ticker, period=period_fallback,
                        interval=interval, auto_adjust=False, progress=False
                    )

                # --- normalizzazione colonne e tz come già fai dopo ---

                # normalizza colonne
                if isinstance(df_all.columns, pd.MultiIndex):
                    df_all.columns = ['_'.join(col).strip() for col in df_all.columns.values]
                    colmap = {}
                    for base in ["Open", "High", "Low", "Close", "Volume"]:
                        matches = [c for c in df_all.columns if c.startswith(base)]
                        if matches: colmap[base] = matches[0]
                    df_all = df_all.rename(columns={v: k for k, v in colmap.items() if v in df_all.columns})

                # to naive index per confronto
                if getattr(df_all.index, "tz", None) is not None:
                    df_all.index = df_all.index.tz_localize(None)
                df_all = df_all.sort_index()

                # Ultima barra entro la GIORNATA As-Of (23:59), robusto per intraday e timezone
                import pandas as pd
                asof_day_end = pd.Timestamp(asof_dt.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                cutoff_idx = df_all.index[df_all.index <= asof_day_end]
                if len(cutoff_idx) == 0:
                    # come fallback estremo, prova anche fino a mezzogiorno del giorno dopo
                    fallback_limit = asof_day_end + pd.Timedelta(hours=12)
                    cutoff_idx = df_all.index[df_all.index <= fallback_limit]
                    if len(cutoff_idx) == 0:
                        raise ValueError("Nessuna barra entro la data As-Of per questo intervallo.")
                last_asof_ts = cutoff_idx[-1]

                # dati fino ad As-Of (con buffer)
                df_past = df_all.loc[:last_asof_ts].tail(days + pre_buffer_days)
                if df_past.shape[0] < 50:
                    raise ValueError("Storico insufficiente prima dell'As-Of.")

                # Stocastico con K auto dello scenario
                k_auto = self._recommended_k(days, interval)
                smooth_k = int(self.smooth_k_var.get() or "3")
                smooth_d = int(self.smooth_d_var.get() or "3")
                _, sk, sd = stochastic(df_past, k_period=k_auto, smooth_k=smooth_k, smooth_d=smooth_d)
                sk = sk.loc[:last_asof_ts];
                sd = sd.loc[:last_asof_ts]

                # Raccomandazione "as of"
                rec = make_recommendations(df_past.loc[:last_asof_ts], sk, sd)

                # Futuro fino a lookahead
                df_future = df_all.loc[last_asof_ts:]
                end_limit = asof_dt + pd.Timedelta(days=lookahead)
                df_eval = df_future[df_future.index <= end_limit]
                if df_eval.shape[0] >= 2:
                    p0 = df_eval['Close'].iloc[0]
                    p_end = df_eval['Close'].iloc[-1]
                    ret_pct = (p_end / p0 - 1.0) * 100.0
                else:
                    ret_pct = float('nan')
#Voto badge
                st = rec.get('short_term', {})
                dir_pred = st.get('action', 'ASTIENITI')
                if dir_pred == "LONG":
                    verdict = "HIT ✅" if (pd.notna(ret_pct) and ret_pct > 0) else "MISS ❌"
                elif dir_pred == "SHORT":
                    verdict = "HIT ✅" if (pd.notna(ret_pct) and ret_pct < 0) else "MISS ❌"
                else:
                    verdict = "N/A (Astenersi)"

                sig = rec.get('signals', {}) or {}
                rk = rec.get('risk', {}) or {}

                # badge
                def _b(k):
                    v = sig.get(k, None)
                    return "—" if not v else v.upper()

                stoch = _b("Stoch");
                rsi_b = _b("RSI");
                macd = _b("MACD");
                hist = _b("MACD_hist")
                trend = _b("Trend");
                adx = _b("ADX");
                obv = _b("OBV")

                # ATR
                atr_level = rk.get("level", "N/D");
                atr_pct = rk.get("value_pct", None)
                atr_txt = f"{atr_level}" + (f" ~ {atr_pct}%" if atr_pct is not None else "")

                # giudizio + voto badge + esito
                def _badge_vote_text(sig):
                    val = {'long': 1, 'short': -1, 'neutral': 0}
                    keys = ["Stoch", "RSI", "MACD", "MACD_hist", "Trend", "OBV"]
                    score = sum(val.get(sig.get(k, 'neutral'), 0) for k in keys)
                    if score >= 3:   return "Maggioranza forte LONG"
                    if score <= -3:  return "Maggioranza forte SHORT"
                    if abs(score) == 2: return "Maggioranza debole"
                    return "Disaccordo/laterale"

                badge_vote = _badge_vote_text(sig)
                first_line = ""
                if hasattr(self, "_compose_judgment_text"):
                    first_line = (self._compose_judgment_text(rec, ref_dt=asof_dt) or "").splitlines()[0]

                # Earnings alert (As-Of)
                pe, ne, dsp, dun, near = earnings_proximity(ticker, asof_dt, window_days=5)
                earn_txt = ""
                if near and ne:
                    earn_txt = f" | ⚠️ Earnings in {dun}g ({ne.strftime('%Y-%m-%d')})"
                elif dsp is not None and dsp <= 2 and pe:
                    earn_txt = f" | ⚠️ Earnings appena usciti ({pe.strftime('%Y-%m-%d')})"

                # Esito
                esito = f"{verdict} ({ret_pct:.2f}%)" if pd.notna(ret_pct) else "Dati futuri insufficienti"

                note = f"{first_line} | Voto badge: {badge_vote}{earn_txt} | {esito}".strip()

                tree.insert("", "end", values=(label, stoch, rsi_b, macd, hist, trend, adx, obv, atr_txt, note))

            except Exception as e:
                tree.insert("", "end", values=(label, "ERR", "—", "—", "—", "—", "—", "—", "—", f"Errore: {e}"))

            pbar['value'] = i
            top.update_idletasks()

    def on_asof_clicked(self):
        self.btn_asof.config(state=tk.DISABLED)
        threading.Thread(target=self._asof_safe, daemon=True).start()

    def _asof_safe(self):
        try:
            self._do_asof_analysis()
        except Exception as e:
            messagebox.showerror("Errore As-Of", str(e))
        finally:
            self.btn_asof.config(state=tk.NORMAL)

    def _do_asof_analysis(self):
        import datetime as dt
        import pandas as pd

        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            raise ValueError("Inserisci un ticker valido.")

        asof_str = (self.asof_var.get() or "").strip()
        if not asof_str:
            raise ValueError("Inserisci una data As-Of nel formato YYYY-MM-DD (es. 2025-06-07).")

        # parse data (naive, senza timezone)
        try:
            asof_dt = dt.datetime.strptime(asof_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Formato data non valido. Usa YYYY-MM-DD.")

        # parametri GUI
        try:
            days = int(self.days_var.get())
            k_period = int(self.k_period_var.get())
            smooth_k = int(self.smooth_k_var.get())
            smooth_d = int(self.smooth_d_var.get())
            lookahead = int(self.lookahead_var.get())
            if min(days, k_period, smooth_k, smooth_d, lookahead) <= 0:
                raise ValueError
        except Exception:
            raise ValueError("Controlla giorni/K/smooth/lookahead: devono essere interi positivi.")

        interval = self.interval_var.get()

        # Yahoo caps per intraday
        def _cap_period(days, interval):
            if interval in ("1m",):
                return f"{min(days, 7)}d"
            elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h", "6h"):
                return f"{min(days, 60)}d"
            else:
                return f"{days}d"

        # buffer per indicatori lunghi (SMA200 & co.)
        pre_buffer_days = max(200, k_period * 4)
        total_days = days + pre_buffer_days + lookahead

        period_all = _cap_period(total_days, interval)
        df_all = yf.download(
            tickers=ticker,
            period=period_all,
            interval=interval,
            auto_adjust=False,
            progress=False
        )
        if df_all is None or df_all.empty:
            raise ValueError("Dati non disponibili per quel range/intervallo.")

        # normalizza colonne se MultiIndex
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = ['_'.join(col).strip() for col in df_all.columns.values]
            colmap = {}
            for base in ["Open", "High", "Low", "Close", "Volume"]:
                matches = [c for c in df_all.columns if c.startswith(base)]
                if matches: colmap[base] = matches[0]
            df_all = df_all.rename(columns={v: k for k, v in colmap.items() if v in df_all.columns})

        # timezone → rende l'indice naive per confrontarlo con asof_dt naive
        if getattr(df_all.index, "tz", None) is not None:
            # to naive (UTC → naive)
            df_all.index = df_all.index.tz_localize(None)

        df_all = df_all.sort_index()

        # trova l'ultima barra <= as-of
        cutoff_idx = df_all.index[df_all.index <= asof_dt]
        if len(cutoff_idx) == 0:
            # se non c'è nulla fino a quell'orario, prova ad allargare al giorno successivo alle 23:59
            fallback_limit = pd.Timestamp(asof_dt) + pd.Timedelta(hours=23, minutes=59)
            cutoff_idx = df_all.index[df_all.index <= fallback_limit]
            if len(cutoff_idx) == 0:
                raise ValueError(
                    "Nessuna barra disponibile entro la data indicata (troppo indietro per questo intervallo).")

        last_asof_ts = cutoff_idx[-1]

        # Dati fino a as-of con buffer
        df_past = df_all.loc[:last_asof_ts].tail(days + pre_buffer_days)
        if df_past.shape[0] < max(k_period + smooth_k + smooth_d + 10, 60):
            raise ValueError("Storico insufficiente prima della data per calcolare indicatori stabili.")

        # Stocastico con i parametri correnti GUI
        _, slow_k, slow_d = stochastic(df_past, k_period=k_period, smooth_k=smooth_k, smooth_d=smooth_d)
        slow_k = slow_k.loc[:last_asof_ts]
        slow_d = slow_d.loc[:last_asof_ts]

        # Raccomandazione "come se fossi lì"
        rec_asof = make_recommendations(df_past.loc[:last_asof_ts], slow_k, slow_d)
        self.last_rec = rec_asof  # utile per candele/giudizio

        # Verifica successiva entro lookahead giorni di calendario
        end_limit = pd.Timestamp(asof_dt) + pd.Timedelta(days=lookahead)
        df_future = df_all.loc[last_asof_ts:]
        df_eval = df_future[df_future.index <= end_limit]
        if df_eval.shape[0] < 2:
            raise ValueError(
                "Poche barre future entro il lookahead per valutare la previsione (aumenta lookahead o cambia intervallo).")

        p0 = df_eval['Close'].iloc[0]
        p_end = df_eval['Close'].iloc[-1]
        ret_pct = (p_end / p0 - 1.0) * 100.0

        st = rec_asof.get('short_term', {})
        dir_pred = st.get('action', 'ASTIENITI')
        if dir_pred == "LONG":
            verdict = "HIT ✅" if ret_pct > 0 else "MISS ❌"
        elif dir_pred == "SHORT":
            verdict = "HIT ✅" if ret_pct < 0 else "MISS ❌"
        else:
            verdict = "N/A (Astenersi)"

        # Mostra risultati
        self._show_asof_result_window(ticker, last_asof_ts, interval, lookahead, rec_asof, ret_pct, verdict, df_eval)

    def _show_asof_result_window(self, ticker, asof_ts, interval, lookahead, rec, ret_pct, verdict, df_eval):
        top = tk.Toplevel(self)
        top.title(f"As-Of {ticker} @ {asof_ts.strftime('%Y-%m-%d %H:%M')} ({interval})")
        top.geometry("900x600")

        # Riepilogo testo
        st = rec.get('short_term', {})
        lt = rec.get('long_term', {})
        risk = rec.get('risk', {})
        lines = []
        lines.append(
            f"As-Of: {asof_ts.strftime('%Y-%m-%d %H:%M')}  |  Intervallo: {interval}  |  Lookahead: {lookahead}g")
        lines.append(
            f"Breve: {st.get('action', '—')} (conf. {st.get('confidence', '—')}, leva {st.get('leverage', '?')})")
        lines.append(f"Motivo breve: {st.get('reason', '—')}")
        lines.append(
            f"Lungo: {lt.get('action', '—')} (conf. {lt.get('confidence', '—')}, leva {lt.get('leverage', '?')})")
        lines.append(f"Rischio (ATR%): {risk.get('level', 'N/D')}"
                     + (f" ~ {risk.get('value_pct')}%" if risk.get('value_pct') is not None else ""))
        lines.append(f"Performance successiva: {ret_pct:.2f}%  →  {verdict}")
        # Avviso earnings riferito all'As-Of
        try:
            asof_naive = asof_ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            asof_naive = dt.datetime.strptime(asof_ts.strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M")
        tkr = self.last_ticker or ticker
        pe, ne, dsp, dun, near = earnings_proximity(tkr, asof_naive, window_days=5)
        if near and ne:
            lines.append(f"Attenzione: earnings in {dun}g alla data As-Of ({ne.strftime('%Y-%m-%d')}).")
        elif dsp is not None and dsp <= 2 and pe:
            lines.append(f"Attenzione: earnings pubblicati da {dsp}g alla data As-Of ({pe.strftime('%Y-%m-%d')}).")

        frm = ttk.Frame(top, padding=8);
        frm.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(frm, wrap="word", height=10)
        txt.pack(fill=tk.X, expand=False)
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

        # Grafico mini della finestra di verifica
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = Figure(figsize=(8.6, 3.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(df_eval.index, df_eval['Close'])
        ax.set_title(f"Prezzo nel lookahead ({lookahead}g) — ritorno {ret_pct:.2f}%")
        ax.grid(True, alpha=0.25)
        canvas = FigureCanvasTkAgg(fig, master=frm)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def on_multi_tf_clicked(self):
        self.btn_multitf.config(state=tk.DISABLED)
        threading.Thread(target=self._multi_tf_check_safe, daemon=True).start()

    def _multi_tf_check_safe(self):
        try:
            self._multi_tf_check()
        except Exception as e:
            messagebox.showerror("Errore Multi-TF", str(e))
        finally:
            self.btn_multitf.config(state=tk.NORMAL)

    def _multi_tf_check(self):
        import datetime as dt

        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            raise ValueError("Inserisci un ticker valido.")

        # Scenari richiesti
        scenarios = [
            ("5g / 5m", 5, "5m"),
            ("15g / 30m", 15, "30m"),
            ("20g / 1h", 20, "1h"),
            ("70g / 1h", 70, "1h"),
            ("100g / 1d", 100, "1d"),
            ("365g / 1wk", 365, "1wk"),
        ]

        # Finestra risultati
        top = tk.Toplevel(self)
        top.title(f"Multi-TF Check — {ticker}")
        top.geometry("1100x500")

        # Header + progress
        hdr = ttk.Frame(top);
        hdr.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(hdr, text=f"Ticker: {ticker}", font=("", 11, "bold")).pack(side=tk.LEFT)
        pbar = ttk.Progressbar(hdr, mode="determinate", maximum=len(scenarios), length=250)
        pbar.pack(side=tk.RIGHT, padx=8)

        # Tabella risultati: badge + ATR + NOTE
        cols = ("scenario", "stoch", "rsi", "macd", "hist", "trend", "adx", "obv", "atr", "note")
        widths = (140, 70, 70, 70, 70, 80, 60, 60, 80, 520)
        headers = ("SCENARIO", "STOCH", "RSI", "MACD", "MACD_H", "TREND", "ADX", "OBV", "ATR", "NOTE")
        tree = ttk.Treeview(top, columns=cols, show="headings", height=20)
        for c, w, h in zip(cols, widths, headers):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="w")
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Scrollbar verticale
        vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.place(in_=tree, relx=1.0, rely=0, relheight=1.0, x=0, y=0, anchor="ne")

        # Scrollbar verticale
        vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.place(in_=tree, relx=1.0, rely=0, relheight=1.0, x=0, y=0, anchor="ne")

        # Helper: download con caps Yahoo (come fai già nel _do_plot)
        def _cap_period(days, interval):
            if interval in ("1m",):
                return f"{min(days, 7)}d"
            elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h"):
                return f"{min(days, 60)}d"
            else:
                return f"{days}d"

        # Loop degli scenari
        for i, (label, days, interval) in enumerate(scenarios, start=1):
            note = []
            try:
                period = _cap_period(days, interval)

                # Auto-K coerente con il tuo algoritmo
                k_auto = self._recommended_k(days, interval)
                smooth_k = int(self.smooth_k_var.get() or "3")
                smooth_d = int(self.smooth_d_var.get() or "3")

                df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, progress=False)
                if df is None or df.empty:
                    raise ValueError("no data")

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    colmap = {}
                    for base in ["Open", "High", "Low", "Close", "Volume"]:
                        matches = [c for c in df.columns if c.startswith(base)]
                        if matches: colmap[base] = matches[0]
                    df = df.rename(columns={v: k for k, v in colmap.items() if v in df.columns})

                # calcolo stocastico
                fast_k, slow_k, slow_d = stochastic(df, k_period=k_auto, smooth_k=smooth_k, smooth_d=smooth_d)

                # raccomandazioni (passa l’atr_period se lo fai scegliere)
                rec = make_recommendations(df, slow_k, slow_d)

                st = rec.get('short_term', {})
                lt = rec.get('long_term', {})
                rk = rec.get('risk', {})

                atr_txt = f"{rk.get('value_pct', '')}%" if rk.get('value_pct') is not None else "N/D"

                # Nota di conflitto/allineamento (facoltativa)
                sig = rec.get('signals', {}) or {}
                trio = [sig.get("Stoch", "neutral"), sig.get("RSI", "neutral"), sig.get("MACD", "neutral")]
                if trio.count("long") >= 2 and sig.get("OBV", "neutral") == "long":
                    note.append("Stoch/RSI/MACD pro-LONG + OBV up")
                elif trio.count("short") >= 2 and sig.get("OBV", "neutral") == "short":
                    note.append("Stoch/RSI/MACD pro-SHORT + OBV down")

                # Inserisci riga
                # --- Calcolo raccomandazioni come prima ---
                rec = make_recommendations(df, slow_k, slow_d)

                sig = rec.get('signals', {}) or {}
                rk = rec.get('risk', {}) or {}

                # Estraggo i badge (default '—' se mancano)
                def _fmt_badge(k):
                    v = sig.get(k, None)
                    if not v:
                        return "—"
                    v = v.upper()
                    return v  # "LONG"/"SHORT"/"NEUTRAL"

                stoch = _fmt_badge("Stoch")
                rsi = _fmt_badge("RSI")
                macd = _fmt_badge("MACD")
                hist = _fmt_badge("MACD_hist")
                trend = _fmt_badge("Trend")
                adx = _fmt_badge("ADX")
                obv = _fmt_badge("OBV")

                # ATR (livello + %, se presente)
                atr_level = rk.get("level", "N/D")
                atr_pct = rk.get("value_pct", None)
                atr_txt = f"{atr_level}" + (f" ~ {atr_pct}%" if atr_pct is not None else "")

                # Calcola voto badge
                vote_val = {'long': 1, 'short': -1, 'neutral': 0}
                keys = ["Stoch", "RSI", "MACD", "MACD_hist", "Trend", "OBV"]
                score = sum(vote_val.get(sig.get(k, 'neutral'), 0) for k in keys)
                if score >= 3:
                    badge_vote = "Maggioranza forte LONG"
                elif score <= -3:
                    badge_vote = "Maggioranza forte SHORT"
                elif abs(score) == 2:
                    badge_vote = "Maggioranza debole"
                else:
                    badge_vote = "Disaccordo/laterale"

                # Giudizio sintetico
                if hasattr(self, "_compose_judgment_text"):
                    jtxt = self._compose_judgment_text(rec)
                    first_line = jtxt.splitlines()[0] if jtxt else ""
                    note = f"{first_line} | Voto badge: {badge_vote}"
                else:
                    note = f"Voto badge: {badge_vote}"

                # Inserisci riga
                tree.insert("", "end", values=(
                    label, stoch, rsi, macd, hist, trend, adx, obv, atr_txt, note
                ))


            except Exception as e:
                tree.insert("", "end", values=(label, "ERR", "—", "—", "—", "—", "—", "—", f"Errore: {e}"))

            pbar['value'] = i
            top.update_idletasks()

    def _open_candles_window(self):
        if self.last_df is None or self.last_df.empty:
            messagebox.showinfo("Info", "Prima genera un grafico: nessun dato caricato.")
            return

        mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        top = tk.Toplevel(self)
        tkr = self.last_ticker or "TICKER"
        top.title(f"Candele — {tkr}")
        top.geometry("920x780")

        # contenitore verticale: grafico sopra, giudizio sotto
        cont = ttk.Frame(top)
        cont.pack(fill=tk.BOTH, expand=True)

        # grafico
        fig, _ = mpf.plot(
            self.last_df,
            type='candle',
            style=s,
            mav=(20, 50),
            volume=True,
            returnfig=True,
            figsize=(9, 6.5),
            tight_layout=True
        )
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=cont)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # separatore
        ttk.Separator(cont, orient="horizontal").pack(fill=tk.X, pady=6)

        # giudizio combinato (usa l'ultimo rec se disponibile)
        text = "Giudizio non disponibile."
        if getattr(self, "last_rec", None):
            text = self._compose_judgment_text(self.last_rec)

        # riquadro scrollabile per il testo (se serve)
        frame_txt = ttk.Frame(cont)
        frame_txt.pack(fill=tk.BOTH, expand=False)
        txt = tk.Text(frame_txt, wrap="word", height=8)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(frame_txt, orient="vertical", command=txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.configure(yscrollcommand=sb.set)

        txt.insert("1.0", text)
        txt.config(state="disabled")

    def _build_plot(self, parent=None):
        parent = parent or self
        self.fig = Figure(figsize=(10, 7.5), dpi=100)
        self.ax_price = self.fig.add_subplot(3, 1, 1)
        self.ax_stoch = self.fig.add_subplot(3, 1, 2)
        self.ax_macd = self.fig.add_subplot(3, 1, 3)  # se non usi MACD, lascia 2 subplot
        self.fig.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def _build_reco_panel(self, parent=None):
        parent = parent or self

        # ===== Scroll container =====
        # Canvas + scrollbar verticale + frame interno
        outer = ttk.Frame(parent)
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.reco_canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.reco_canvas.yview)
        self.reco_canvas.configure(yscrollcommand=vscroll.set)

        self.reco_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # frame interno (il "vero" pannello consigli)
        self.reco_inner = ttk.Frame(self.reco_canvas, padding=(10, 6, 10, 6))
        self.reco_window = self.reco_canvas.create_window(
            (0, 0), window=self.reco_inner, anchor="nw"
        )

        # --- sync altezza/ampiezza e area scrollabile
        def _on_inner_config(event=None):
            self.reco_canvas.configure(scrollregion=self.reco_canvas.bbox("all"))

        self.reco_inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(event):
            # forza il frame interno ad avere la stessa larghezza del canvas
            self.reco_canvas.itemconfigure(self.reco_window, width=event.width)

        self.reco_canvas.bind("<Configure>", _on_canvas_config)

        # --- mouse wheel (Windows/Mac/Linux)
        def _bind_wheel(widget):
            # Windows / Linux
            widget.bind_all("<MouseWheel>", lambda e: self.reco_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
            # macOS
            widget.bind_all("<Button-4>", lambda e: self.reco_canvas.yview_scroll(-1, "units"))
            widget.bind_all("<Button-5>", lambda e: self.reco_canvas.yview_scroll(1, "units"))

        _bind_wheel(self.reco_canvas)

        # ====== CONTENUTO (come prima, ma nel self.reco_inner) ======
        # Titolo
        ttk.Label(
            self.reco_inner,
            text="Consigli automatici (sperimentale)",
            font=("", 11, "bold")
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        # BADGE (2 righe auto-layout)
        badges = ttk.Frame(self.reco_inner)
        badges.grid(row=1, column=0, columnspan=6, sticky="w", pady=(0, 8))
        self.badge_labels = {}

        badge_keys = ["RSI", "MACD", "MACD_hist", "Stoch", "Trend", "ATR", "ADX", "OBV"]
        MAX_COLS = 6
        for i, key in enumerate(badge_keys):
            r = i // MAX_COLS
            c = i % MAX_COLS
            lbl = tk.Label(badges, text=f"{key}: —", padx=8, pady=2, bd=1, relief="groove")
            lbl.grid(row=r, column=c, padx=4, pady=2, sticky="w")
            self.badge_labels[key] = lbl

        # Breve
        self.short_label = ttk.Label(self.reco_inner, text="Breve periodo (1–7g): —", font=("", 10, "bold"))
        self.short_label.grid(row=2, column=0, sticky="w")
        self.short_reason = ttk.Label(self.reco_inner, text="Motivo: —", wraplength=1000, justify="left")
        self.short_reason.grid(row=3, column=0, columnspan=6, sticky="w")

        # Lungo
        self.long_label = ttk.Label(self.reco_inner, text="Periodo più lungo (swing): —", font=("", 10, "bold"))
        self.long_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.long_reason = ttk.Label(self.reco_inner, text="Motivo: —", wraplength=1000, justify="left")
        self.long_reason.grid(row=5, column=0, columnspan=6, sticky="w")

        # Rischio / ATR%
        self.risk_label = ttk.Label(self.reco_inner, text="Rischio (ATR%): —")
        self.risk_label.grid(row=6, column=0, sticky="w", pady=(6, 0))
        # Giudizio combinato
        ttk.Label(self.reco_inner, text="Giudizio combinato", font=("", 10, "bold")).grid(
            row=8, column=0, sticky="w", pady=(10, 2)
        )
        self.judgment_label = ttk.Label(self.reco_inner, text="—", wraplength=1000, justify="left")
        self.judgment_label.grid(row=9, column=0, columnspan=6, sticky="w")

        # Bottone legenda (se vuoi)
        ttk.Button(self.reco_inner, text="Apri legenda", command=self.show_legenda).grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        # Dimensione posizione (Kelly½)
        self.size_short_label = ttk.Label(self.reco_inner, text="Size breve (Kelly½): —")
        self.size_short_label.grid(row=7, column=0, sticky="w", pady=(6, 0))

        self.size_long_label = ttk.Label(self.reco_inner, text="Size lungo (Kelly½): —")
        self.size_long_label.grid(row=7, column=1, sticky="w", pady=(6, 0))
        # colonna elastica
        self.reco_inner.grid_columnconfigure(0, weight=1)

    def _on_resize(self, event):
        # calcola wrap ~80% della larghezza attuale
        wl = max(400, int(self.winfo_width() * 0.8))
        try:
            self.short_reason.config(wraplength=wl)
            self.long_reason.config(wraplength=wl)
        except Exception:
            pass
    def on_plot_clicked(self):
        self.btn.config(state=tk.DISABLED)
        threading.Thread(target=self._plot_safe, daemon=True).start()

    def _plot_safe(self):
        try:
            self._do_plot()
        except Exception as e:
            messagebox.showerror("Errore", str(e))
        finally:
            self.btn.config(state=tk.NORMAL)

    def _parse_ticker(self, raw: str) -> str:
        if not raw:
            return ""
        return raw.strip().split()[0].upper()

    def _do_plot(self):
        ticker = self._parse_ticker(self.ticker_var.get())
        if not ticker:
            raise ValueError("Inserisci un ticker valido, es. 'TSLA'.")

        # --- giorni ---
        try:
            days = int(self.days_var.get())
            if days <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("I giorni devono essere un intero positivo, es. 30.")

        # --- interval ---
        interval = self.interval_var.get()

        # >>>>>>>  PRIMA di leggere K: applica Auto-K se attivo  <<<<<<<
        if self.auto_k_var.get():
            k_auto = self._recommended_k(days, interval)
            self.k_period_var.set(str(k_auto))

        # --- ora leggi/valida K e smoothing ---
        try:
            k_period = int(self.k_period_var.get())
            smooth_k = int(self.smooth_k_var.get())
            smooth_d = int(self.smooth_d_var.get())
            if min(k_period, smooth_k, smooth_d) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Parametri stocastici devono essere interi positivi.")

        if self.auto_k_var.get():
            k_auto = self._recommended_k(days, interval)
            self.k_period_var.set(str(k_auto))
        period = f"{days}d"
        if interval in ("1m",):
            period = f"{min(days, 7)}d"
        elif interval in ("2m", "5m", "15m", "30m", "60m", "90m", "1h"):
            period = f"{min(days, 60)}d"

        df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise ValueError(f"Nessun dato trovato per '{ticker}' (controlla ticker/intervallo).")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            colmap = {}
            for base in ["Open", "High", "Low", "Close", "Volume"]:
                matches = [c for c in df.columns if c.startswith(base)]
                if matches:
                    colmap[base] = matches[0]
            df = df.rename(columns={v: k for k, v in colmap.items() if v in df.columns})

        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            raise ValueError("Le colonne OHLC non sono disponibili per il dataset scaricato.")

        fast_k, slow_k, slow_d = stochastic(df, k_period=k_period, smooth_k=smooth_k, smooth_d=smooth_d)



        # ---- RSI & MACD ----
        rsi14 = rsi(df['Close'], 14)
        macd_line, macd_signal, macd_hist = macd(df['Close'])
        adx_val, plus_di, minus_di = adx(df, n=14)

        # OBV
        obv_val = obv(df)
        self.last_df = df.copy()
        self.last_ticker = ticker
        # ---- Grafici ----
        # Prezzo
        self.ax_price.clear()
        self.ax_price.plot(df.index, df['Close'], label=f"{ticker} Close")
        self.ax_price.set_title(f"{ticker} - Andamento ultimi {days}g (interval {interval})")
        self.ax_price.set_ylabel("Prezzo")
        self.ax_price.grid(True, alpha=0.25)
        self.ax_price.legend(loc="upper left")

        # Stocastico + RSI
        self.ax_stoch.clear()
        self.ax_stoch.plot(df.index, fast_k, label="Fast %K", linewidth=1)
        self.ax_stoch.plot(df.index, slow_k, label="Slow %K", linewidth=1.6)
        self.ax_stoch.plot(df.index, slow_d, label="Slow %D", linewidth=1.6, linestyle="--")
        self.ax_stoch.axhline(80, linestyle="--", linewidth=1)
        self.ax_stoch.axhline(20, linestyle="--", linewidth=1)
        # RSI sovrapposto (scala 0-100)
        self.ax_stoch.plot(df.index, rsi14, label="RSI(14)", linewidth=1)
        self.ax_stoch.axhline(70, linestyle=":", linewidth=1)
        self.ax_stoch.axhline(30, linestyle=":", linewidth=1)
        self.ax_stoch.set_ylim(0, 100)
        self.ax_stoch.set_ylabel("Stocastico / RSI")
        self.ax_stoch.grid(True, alpha=0.25)
        self.ax_stoch.legend(loc="upper left", ncols=3)

        # MACD
        self.ax_macd.clear()
        # istogramma
        self.ax_macd.bar(df.index, macd_hist, width=0.8, align='center')
        # linee
        self.ax_macd.plot(df.index, macd_line, linewidth=1, label="MACD")
        self.ax_macd.plot(df.index, macd_signal, linewidth=1, linestyle="--", label="Signal")
        self.ax_macd.axhline(0, linewidth=1)
        self.ax_macd.set_ylabel("MACD")
        self.ax_macd.set_xlabel("Data")
        self.ax_macd.grid(True, alpha=0.25)
        self.ax_macd.legend(loc="upper left")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        last = df.index[-1]
        self.title(f"Grafico Azione + Stocastico — {ticker} (ultimo: {last.strftime('%Y-%m-%d %H:%M')})")
        # Aggiorna barra stato con timezone exchange + locale
        self.status_var.set(self._format_last_time_text(last))

        # ---- Consigli ----
        # ---- Consigli ----
        atr_period = int(self.atr_period_var.get())
        rec = make_recommendations(df, slow_k, slow_d, atr_period=atr_period)
        # Dopo aver impostato short_label / long_label / risk_label ...
        st = rec.get('short_term', {})
        lt = rec.get('long_term', {})

        try:
            ks = float(st.get('kelly_half', 0.0))
            self.size_short_label.config(text=f"Size breve (Kelly½): {ks*100:.1f}% del capitale")
        except Exception:
            self.size_short_label.config(text="Size breve (Kelly½): —")

        try:
            kl = float(lt.get('kelly_half', 0.0))
            self.size_long_label.config(text=f"Size lungo (Kelly½): {kl*100:.1f}% del capitale")
        except Exception:
            self.size_long_label.config(text="Size lungo (Kelly½): —")

        self._update_reco_panel(rec)
        self.last_rec = rec  # memorizza per la finestra candele

    def show_legenda(self):
        """Apre una finestra con la legenda dei badge"""
        top = tk.Toplevel(self)
        top.title("Legenda Indicatori")
        top.geometry("700x600")

        # Box di testo scrollabile
        txt = tk.Text(top, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=10, pady=10)

        scroll = ttk.Scrollbar(top, command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        # Carica il file legenda (se esiste)
        try:
            with open("C:/Users/PC/Desktop/legenda_badge.txt", "r", encoding="utf-8") as f:
                legenda_text = f.read()
        except Exception as e:
            legenda_text = f"Errore nel caricamento della legenda:\n{e}"

        txt.insert("1.0", legenda_text)
        txt.config(state="disabled")  # sola lettura
    def _update_reco_panel(self, rec):
        if not isinstance(rec, dict):
            self.short_label.config(text="Breve periodo (1–7g): —")
            self.short_reason.config(text="Motivo: Consiglio non disponibile")
            self.long_label.config(text="Periodo più lungo (swing): —")
            self.long_reason.config(text="Motivo: Consiglio non disponibile")
            self.risk_label.config(text="Rischio (ATR%): —")
            return

        st = rec.get('short_term', {}) or {}
        lt = rec.get('long_term', {}) or {}
        rk = rec.get('risk', {}) or {}

        # Breve periodo
        st = rec['short_term']
        lev_st = st.get('leverage', '?')
        if isinstance(lev_st, float):
            lev_st_str = f"{lev_st:.0f}"
        else:
            lev_st_str = str(lev_st)
        self.short_label.config(
            text=f"Breve periodo (1–7g): {st['action']} (confidenza {st['confidence']}, leva {lev_st_str})"
        )
        self.short_reason.config(text=f"Motivo: {st['reason']}")

        # Lungo periodo
        lt = rec['long_term']
        lev_lt = lt.get('leverage', '?')
        if isinstance(lev_lt, (int, float)):
            lev_lt_str = f"{float(lev_lt):.1f}"
        else:
            lev_lt_str = str(lev_lt)
        self.long_label.config(
            text=f"Periodo più lungo (swing): {lt['action']} (confidenza {lt['confidence']}, leva {lev_lt_str})"
        )
        self.long_reason.config(text=f"Motivo: {lt['reason']}")

        # Rischio
        rk = rec['risk']
        atr_p = self.atr_period_var.get()
        if rk['value_pct'] is not None:
            self.risk_label.config(text=f"Rischio (ATR{atr_p} % prezzo): {rk['level']} ~ {rk['value_pct']}%")
        else:
            self.risk_label.config(text=f"Rischio (ATR{atr_p} % prezzo): {rk['level']}")
        # Badge colorati
        signals = rec.get('signals', {}) or {}
        for key, lbl in self.badge_labels.items():
            state = signals.get(key, 'neutral')
            colors = self._sig_colors.get(state, self._sig_colors['neutral'])

            # testo di default
            txt = f"{key}: {state.upper()}" if key in signals else f"{key}: —"

            # se è ADX e hai anche il valore numerico, mostralo
            if key == "ADX" and 'ADX_value' in signals and signals['ADX_value'] is not None:
                txt = f"ADX {signals['ADX_value']}: {state.upper()}"

            lbl.config(text=txt, fg=colors['fg'], bg=colors['bg'])
        self._update_judgment(rec)

    def _infer_horizon(self) -> str:
        """
        Restituisce 'breve' | 'medio' | 'lungo' in base a preset se presente,
        altrimenti da (days, interval).
        """
        # Se hai self.preset_var dalla Combobox dei preset
        preset = getattr(self, "preset_var", None)
        if preset:
            p = preset.get()
            if p == "Breve": return "breve"
            if p == "Medio": return "medio"
            if p == "Lungo": return "lungo"

        # Heuristica se non usi i preset
        try:
            days = int(self.days_var.get())
        except Exception:
            days = 30
        interval = self.interval_var.get()

        intraday = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "6h")
        if intraday:
            if days <= 30:  return "breve"
            if days <= 120: return "medio"
            return "lungo"
        if interval in ("1d",):
            if days <= 60:  return "breve"
            if days <= 300: return "medio"
            return "lungo"
        if interval in ("1wk", "1mo"):
            return "lungo"
        return "medio"

    def _update_judgment(self, rec: dict):
        """
        Costruisce un giudizio in linguaggio naturale combinando i badge
        e l'ATR, considerando l'orizzonte (breve/medio/lungo).
        """
        try:
            signals = rec.get('signals', {}) or {}
            risk = rec.get('risk', {}) or {}
            st = rec.get('short_term', {}) or {}
            lt = rec.get('long_term', {}) or {}

            horizon = self._infer_horizon()

            # Voto dei principali segnali direzionali
            vote_val = {'long': 1, 'short': -1, 'neutral': 0}
            keys = ["Stoch", "RSI", "MACD", "MACD_hist", "Trend", "OBV"]
            votes = [vote_val.get(signals.get(k, 'neutral'), 0) for k in keys]
            score = sum(votes)  # da -6 a +6

            # Forza trend da ADX (se disponibile e >25)
            adx_state = signals.get("ADX", "neutral")
            adx_text = None
            if "ADX_value" in signals and signals["ADX_value"] is not None:
                adx_text = f"ADX={signals['ADX_value']}"
            trend_strong = (adx_state in ("long", "short"))

            # Rischio ATR
            atr_level = risk.get("level", "N/D")
            atr_pct = risk.get("value_pct", None)

            lines = []

            # Testa primo: orizzonte
            if horizon == "breve":
                # Lo stocastico comanda
                base_dir = st.get("action", "ASTIENITI")
                base_conf = st.get("confidence", "bassa")
                if base_dir == "ASTIENITI":
                    lines.append("Breve periodo: **Astenersi** — segnale principale debole/contrasto.")
                else:
                    lines.append(f"Breve periodo: **{base_dir}** (conf. {base_conf}).")
            elif horizon == "medio":
                # equilibrio tra stocastico e trend
                base_dir = st.get("action", "ASTIENITI")
                trend = signals.get("Trend", "neutral")
                if base_dir != "ASTIENITI" and trend != "neutral":
                    if (base_dir == "LONG" and trend == "long") or (base_dir == "SHORT" and trend == "short"):
                        lines.append(f"Medio periodo: **{base_dir}** — stocastico allineato al trend (SMA200).")
                    else:
                        lines.append("Medio periodo: **prudenza** — stocastico contro trend (possibile pullback).")
                else:
                    lines.append("Medio periodo: **neutro/prudenza** — segnali misti.")
            else:  # lungo
                base_dir = lt.get("action", "ASTIENITI")
                base_conf = lt.get("confidence", "bassa")
                if base_dir == "ASTIENITI":
                    lines.append("Lungo periodo: **Astenersi** — prezzo vicino a SMA200 o segnali deboli.")
                else:
                    lines.append(f"Lungo periodo: **{base_dir}** (conf. {base_conf}).")

            # Regole interpretative extra (come la tua lista)
            # 1) Stoch + RSI + MACD long & OBV long => segnale forte LONG
            trio = [signals.get("Stoch", "neutral"), signals.get("RSI", "neutral"), signals.get("MACD", "neutral")]
            if trio.count("long") >= 2 and signals.get("OBV", "neutral") == "long":
                lines.append("Combinazione: **Stoch/RSI/MACD pro-LONG + OBV in aumento** ⇒ segnale **forte LONG**.")
            if trio.count("short") >= 2 and signals.get("OBV", "neutral") == "short":
                lines.append("Combinazione: **Stoch/RSI/MACD pro-SHORT + OBV in calo** ⇒ segnale **forte SHORT**.")

            # 2) Stoch SHORT ma ADX <25 ⇒ trend debole ⇒ prudenza
            if signals.get("Stoch") == "short" and not trend_strong:
                lines.append("Nota: **Stoch SHORT ma ADX basso** ⇒ trend debole/laterale → **prudenza**.")

            # 3) Trend LONG ma Stoch SHORT ⇒ probabile pullback
            if signals.get("Trend") == "long" and signals.get("Stoch") == "short":
                lines.append("Divergenza: **Trend LONG ma Stoch SHORT** ⇒ **pullback probabile** (contro-trend).")
            if signals.get("Trend") == "short" and signals.get("Stoch") == "long":
                lines.append("Divergenza: **Trend SHORT ma Stoch LONG** ⇒ **rimbalzo tecnico** possibile.")

            # 4) ATR alto ⇒ leva ridotta / affidabilità minore
            if atr_level == "ALTO":
                extra = f" (~{atr_pct}%)" if atr_pct is not None else ""
                lines.append(f"Rischio: **ATR ALTO{extra}** ⇒ segnali meno affidabili, **ridurre la leva**.")
            elif atr_level == "BASSO":
                extra = f" (~{atr_pct}%)" if atr_pct is not None else ""
                lines.append(f"Rischio: **ATR BASSO{extra}** ⇒ contesto più stabile (ma occhio ai falsi segnali).")

            # 5) ADX forte ⇒ trend robusto
            if trend_strong:
                dir_txt = "rialzista" if adx_state == "long" else "ribassista"
                if adx_text:
                    lines.append(f"Forza trend: **{adx_text} (>25)** ⇒ trend {dir_txt} **robusto**.")
                else:
                    lines.append(f"Forza trend: **ADX >25** ⇒ trend {dir_txt} **robusto**.")

            # Riassunto voto
            if score >= 3:
                lines.append("Voto badge: **maggioranza forte LONG**.")
            elif score <= -3:
                lines.append("Voto badge: **maggioranza forte SHORT**.")
            elif abs(score) == 2:
                lines.append("Voto badge: **maggioranza debole** (segnali quasi bilanciati).")
            else:
                lines.append("Voto badge: **disaccordo/laterale**.")

            self.judgment_label.config(text="\n".join(lines))
        except Exception as e:
            self.judgment_label.config(text=f"Giudizio non disponibile: {e}")


if __name__ == "__main__":
    app = StochasticApp()
    app.mainloop()
