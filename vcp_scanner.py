"""
VCP Scanner — screaminge.co.uk
Runs every 30 minutes while the US market is open.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import logging
import json
from datetime import datetime, timezone, timedelta
from ftplib import FTP
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────

BATCH_SIZE          = 100
MIN_WEEKS           = 30           # relaxed from 40
MAX_PIVOT_DISTANCE  = 0.15         # relaxed from 0.08 — within 15% of pivot
MIN_DOLLAR_VOLUME   = 1_000_000    # relaxed from 20M — catches mid/small caps
SCAN_INTERVAL_MINS  = 30

TELEGRAM_BOT_TOKEN  = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID    = "YOUR_CHAT_ID"

FTP_HOST            = "ftp.screaminge.co.uk"
FTP_USER            = "YOUR_FTP_USER"
FTP_PASS            = "YOUR_FTP_PASS"
FTP_REMOTE_DIR      = "htdocs"

TICKER_FILES = [
    ("nasdaqlisted.txt", "|", "ETF",  "N", "Symbol"),
    ("otherlisted.txt",  "|", "ETF",  "N", "ACT Symbol"),
]

EXTRA_SYMBOLS = [
    "SHOP.TO","RY.TO","TD.TO","ENB.TO","CP.TO",
    "CNQ.TO","SU.TO","BNS.TO","TRP.TO","BAM.TO",
    "SHEL.L","AZN.L","HSBA.L","BP.L","ULVR.L",
    "RIO.L","BARC.L","DGE.L","GSK.L","REL.L",
]

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  MARKET HOURS
# ─────────────────────────────────────────

def market_is_open():
    et  = ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


def seconds_until_open():
    et        = ZoneInfo("America/New_York")
    now       = datetime.now(et)
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return max(0, int((candidate - now).total_seconds()))

# ─────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────

def send_telegram(message):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10
        )
        if not resp.ok:
            log.warning("Telegram error: %s", resp.text)
    except Exception as e:
        log.warning("Telegram failed: %s", e)

# ─────────────────────────────────────────
#  SYMBOL LIST
# ─────────────────────────────────────────

def get_symbols():
    symbols = list(EXTRA_SYMBOLS)
    for filename, sep, etf_col, etf_val, sym_col in TICKER_FILES:
        try:
            df   = pd.read_csv(filename, sep=sep)
            mask = df[etf_col] == etf_val
            if "Test Issue" in df.columns:
                mask &= df["Test Issue"] == "N"
            symbols += df.loc[mask, sym_col].dropna().tolist()
        except FileNotFoundError:
            log.warning("Ticker file not found: %s — skipping", filename)
    symbols = list({s for s in symbols if isinstance(s, str) and 1 <= len(s) <= 8})
    log.info("Symbol universe: %d tickers", len(symbols))
    return symbols

# ─────────────────────────────────────────
#  BENCHMARK
# ─────────────────────────────────────────

def load_spy():
    spy = yf.download("SPY", period="1y", interval="1wk", progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy

# ─────────────────────────────────────────
#  EXTRACT SINGLE TICKER FROM BATCH
# ─────────────────────────────────────────

def extract_ticker(raw, symbol, batch):
    try:
        if len(batch) == 1:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        if not isinstance(raw.columns, pd.MultiIndex):
            return None
        top = raw.columns.get_level_values(0)
        if symbol not in top:
            return None
        df = raw[symbol].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

# ─────────────────────────────────────────
#  FILTERS
# ─────────────────────────────────────────

def relative_strength(df, spy):
    try:
        sr = float(df["Close"].iloc[-1])  / float(df["Close"].iloc[-26])
        br = float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-26])
        return sr / br if br else 0.0
    except Exception:
        return 0.0


def liquidity_ok(df):
    try:
        dollar_vol = (df["Close"] * df["Volume"]).tail(20).mean()
        return float(dollar_vol) > MIN_DOLLAR_VOLUME
    except Exception:
        return False


def stage2(df):
    """
    Stock must be in a Stage 2 uptrend:
    price > 10wk MA > 30wk MA, and 30wk MA trending up.
    Relaxed from 40wk to 30wk MA.
    """
    try:
        close = df["Close"]
        ma10  = close.rolling(10).mean()
        ma30  = close.rolling(30).mean()
        if ma30.dropna().shape[0] < 20:
            return False
        return (
            float(close.iloc[-1]) > float(ma10.iloc[-1])
            and float(ma10.iloc[-1]) > float(ma30.iloc[-1])
            and float(ma30.iloc[-1]) > float(ma30.iloc[-6])
        )
    except Exception:
        return False


def prior_uptrend(df):
    """At least 15% move off lows in past 26 weeks. Relaxed from 20%."""
    try:
        closes = df["Close"].tail(26)
        low    = float(closes.min())
        high   = float(closes.max())
        return (high - low) / low > 0.15 if low > 0 else False
    except Exception:
        return False


def detect_vcp(df):
    """
    Core VCP detection — looks for volatility contraction over recent weeks.

    Method: split the last 15 weeks into three 5-week windows.
    Each window's high-low range should be smaller than the previous.
    Requires at least 2 of the 3 contractions to be valid (more lenient).
    Also checks that volume is declining across windows.
    """
    try:
        if len(df) < 15:
            return False, 0, None

        bars = df.tail(15)

        # Split into three 5-week windows
        w1 = bars.iloc[0:5]
        w2 = bars.iloc[5:10]
        w3 = bars.iloc[10:15]

        r1 = float(w1["High"].max() - w1["Low"].min())
        r2 = float(w2["High"].max() - w2["Low"].min())
        r3 = float(w3["High"].max() - w3["Low"].min())

        v1 = float(w1["Volume"].mean())
        v2 = float(w2["Volume"].mean())
        v3 = float(w3["Volume"].mean())

        # Count range contractions (each window tighter than previous)
        range_contractions = sum([r2 < r1, r3 < r2])
        vol_contractions   = sum([v2 < v1, v3 < v2])

        # Need at least 1 range contraction and 1 volume contraction
        if range_contractions < 1 or vol_contractions < 1:
            return False, 0, None

        # Pivot = highest high in the last 10 weeks
        pivot        = float(df["High"].tail(10).max())
        contractions = range_contractions + vol_contractions

        return True, contractions, pivot

    except Exception:
        return False, 0, None


def tightening(df):
    """
    Recent closes should be getting tighter (lower standard deviation).
    Compare std of last 4 weeks vs prior 8 weeks.
    """
    try:
        recent = df["Close"].tail(4)
        prior  = df["Close"].tail(12).head(8)
        recent_std = float(recent.std() / recent.mean())
        prior_std  = float(prior.std()  / prior.mean())
        return recent_std < prior_std
    except Exception:
        return False


def volume_dryup(df):
    """Recent 3-week volume below the prior 9-week average."""
    try:
        recent   = float(df["Volume"].tail(3).mean())
        previous = float(df["Volume"].tail(12).head(9).mean())
        return recent < previous * 0.9 if previous > 0 else False
    except Exception:
        return False


def is_breaking_out(df, pivot):
    """Latest week closed above pivot on above-average volume."""
    try:
        latest  = df.iloc[-1]
        avg_vol = float(df["Volume"].tail(10).mean())
        return (
            float(latest["Close"]) > pivot
            and float(latest["Volume"]) > avg_vol * 1.3
        )
    except Exception:
        return False

# ─────────────────────────────────────────
#  SCORING  (0–100 scale)
# ─────────────────────────────────────────

def score_setup(df, spy, contractions, pct_from_pivot):
    rs         = relative_strength(df, spy)
    vol_score  = 15 if volume_dryup(df)  else 0
    tight_score= 15 if tightening(df)    else 0
    rs_score   = min(rs * 20, 30)        # cap at 30
    prox_score = max(0, (1 - pct_from_pivot / MAX_PIVOT_DISTANCE)) * 25
    cont_score = min(contractions * 5, 15)
    return round(rs_score + prox_score + cont_score + vol_score + tight_score, 1)

# ─────────────────────────────────────────
#  HTML + JSON OUTPUT
# ─────────────────────────────────────────

def export_html(results, scanned):
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = ""
    for rank, r in enumerate(results, 1):
        ticker = r["Ticker"]
        badge  = ' <span style="color:#00ff88;font-weight:bold">⚡ BREAKOUT</span>' if r.get("Breakout") else ""
        chart  = (
            f'<iframe width="420" height="260" '
            f'src="https://s.tradingview.com/widgetembed/?symbol={ticker}'
            f'&interval=W&theme=dark" frameborder="0"></iframe>'
        )
        rows += (
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{ticker}</b>{badge}</td>"
            f"<td>{r['Score']}</td>"
            f"<td>{r['Contractions']}</td>"
            f"<td>{r['PctFromPivot']}%</td>"
            f"<td>{r['Pivot']}</td>"
            f"<td>{chart}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VCP Scanner — screaminge.co.uk</title>
<style>
  body  {{ background:#111; color:#eee; font-family:Arial,sans-serif; padding:20px; }}
  h2   {{ color:#fff; }}
  table{{ border-collapse:collapse; width:100%; }}
  th   {{ background:#222; padding:10px; text-align:left; border:1px solid #333; }}
  td   {{ padding:8px; border:1px solid #333; vertical-align:top; }}
  tr:nth-child(even) {{ background:#1a1a1a; }}
  .meta {{ color:#aaa; margin-bottom:16px; }}
</style>
</head>
<body>
<h2>Screaminge VCP Scanner</h2>
<p class="meta">
  <b>Last scan:</b> {now} &nbsp;|&nbsp;
  <b>Scanned:</b> {scanned} &nbsp;|&nbsp;
  <b>Candidates:</b> {len(results)}
</p>
<table>
<thead>
<tr><th>#</th><th>Ticker</th><th>Score</th><th>Contractions</th><th>% From Pivot</th><th>Pivot</th><th>Chart</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>"""

    with open("results.html", "w", encoding="utf-8") as f:
        f.write(html)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"last_scan": now, "scanned": scanned, "results": results}, f, indent=2)

    log.info("Exported %d candidates", len(results))

# ─────────────────────────────────────────
#  FTP UPLOAD
# ─────────────────────────────────────────

def upload_results():
    try:
        ftp = FTP(FTP_HOST, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)
        ftp.cwd(FTP_REMOTE_DIR)
        with open("results.html", "rb") as f:
            ftp.storbinary("STOR results.html", f)
        with open("results.json", "rb") as f:
            ftp.storbinary("STOR results.json", f)
        ftp.quit()
        log.info("FTP upload complete")
    except Exception as e:
        log.error("FTP upload failed: %s", e)

# ─────────────────────────────────────────
#  CORE SCAN
# ─────────────────────────────────────────

def run_scan(symbols, spy):
    results         = []
    scanned         = 0
    breakout_alerts = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        log.info("Batch %d–%d of %d ...", i + 1, i + len(batch), len(symbols))

        try:
            raw = yf.download(
                tickers=batch,
                period="1y",
                interval="1wk",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            log.warning("Batch download failed: %s", e)
            time.sleep(5)
            continue

        time.sleep(1)

        for symbol in batch:
            scanned += 1
            try:
                df = extract_ticker(raw, symbol, batch)
                if df is None:
                    continue
                df = df.dropna(subset=["Close", "Volume"])
            except Exception:
                continue

            if len(df) < MIN_WEEKS:      continue
            if not liquidity_ok(df):     continue
            if not stage2(df):           continue
            if not prior_uptrend(df):    continue

            is_vcp, contractions, pivot = detect_vcp(df)
            if not is_vcp:               continue

            close          = float(df["Close"].iloc[-1])
            pct_from_pivot = (pivot - close) / pivot if pivot > 0 else 1.0

            if pct_from_pivot > MAX_PIVOT_DISTANCE:
                continue

            score    = score_setup(df, spy, contractions, pct_from_pivot)
            breakout = is_breaking_out(df, pivot)

            entry = {
                "Ticker":       symbol,
                "Score":        score,
                "Contractions": contractions,
                "PctFromPivot": round(pct_from_pivot * 100, 2),
                "Pivot":        round(pivot, 2),
                "Breakout":     breakout,
            }
            results.append(entry)

            if breakout:
                breakout_alerts.append(entry)
                log.info("  ⚡ BREAKOUT  %s  pivot=%.2f  score=%.1f", symbol, pivot, score)
            else:
                log.info("  VCP forming  %s  score=%.1f  %.1f%% from pivot",
                         symbol, score, pct_from_pivot * 100)

    # ── Telegram ─────────────────────────
    if breakout_alerts:
        lines = ["🚨 VCP BREAKOUT ALERT\n"]
        for r in breakout_alerts:
            lines.append(
                f"• {r['Ticker']}  pivot={r['Pivot']}  "
                f"score={r['Score']}  ({r['PctFromPivot']}% from pivot)"
            )
        lines.append(f"\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        send_telegram("\n".join(lines))

    if results:
        top   = sorted(results, key=lambda x: x["Score"], reverse=True)[:8]
        lines = [f"📊 Scan complete — {len(results)} VCP setups forming\n"]
        for r in top:
            flag = "⚡ " if r.get("Breakout") else ""
            lines.append(
                f"{flag}• {r['Ticker']}  score={r['Score']}  "
                f"({r['PctFromPivot']}% from pivot)"
            )
        send_telegram("\n".join(lines))
    else:
        send_telegram(
            f"📊 Scan complete — 0 setups found.\n"
            f"Scanned {scanned} stocks.\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )

    results.sort(key=lambda x: x["Score"], reverse=True)
    export_html(results, scanned)
    upload_results()
    return results

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("VCP Scanner starting up")
    log.info("=" * 50)

    symbols = get_symbols()

    while True:
        if not market_is_open():
            wait = seconds_until_open()
            log.info("Market closed. Next open in %dh %dm.",
                     wait // 3600, (wait % 3600) // 60)
            time.sleep(min(wait, 3600))
            continue

        log.info("Market open — scanning %d symbols", len(symbols))

        try:
            spy = load_spy()
        except Exception as e:
            log.error("Could not load SPY: %s — retrying in 5 min", e)
            time.sleep(300)
            continue

        try:
            results = run_scan(symbols, spy)
            log.info("Scan done. %d VCP candidates.", len(results))
        except Exception as e:
            log.exception("Scan crashed: %s", e)

        log.info("Next scan in %d minutes.", SCAN_INTERVAL_MINS)
        time.sleep(SCAN_INTERVAL_MINS * 60)


if __name__ == "__main__":
    main()
