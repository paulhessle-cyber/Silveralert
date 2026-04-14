"""
VCP Scanner — screaminge.co.uk
Runs every 30 minutes while the US market is open.
Sends Telegram alerts on breakouts and uploads results.html via FTP.
"""

import yfinance as yf
import pandas as pd
import time
import requests
import logging
import json
from datetime import datetime, timezone, timedelta
from ftplib import FTP
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────
#  SETTINGS  — fill these in
# ─────────────────────────────────────────

BATCH_SIZE          = 100
MIN_WEEKS           = 40
MAX_PIVOT_DISTANCE  = 0.08
MIN_DOLLAR_VOLUME   = 20_000_000
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
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def seconds_until_open():
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
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
#  HELPERS — extract a clean single-ticker DataFrame from a batch download
# ─────────────────────────────────────────

def extract_ticker(raw, symbol, batch):
    if len(batch) == 1:
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        df = raw[symbol].copy()
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        return df
    return None

# ─────────────────────────────────────────
#  FILTERS
# ─────────────────────────────────────────

def relative_strength(df, spy):
    try:
        stock_ret = float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-26])
        spy_ret   = float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-26])
        return stock_ret / spy_ret if spy_ret else 0.0
    except Exception:
        return 0.0


def liquidity_ok(df):
    dollar_vol = (df["Close"] * df["Volume"]).tail(20).mean()
    return float(dollar_vol) > MIN_DOLLAR_VOLUME


def stage2(df):
    close = df["Close"]
    ma10  = close.rolling(10).mean()
    ma40  = close.rolling(40).mean()
    if ma40.dropna().shape[0] < 40:
        return False
    return (
        float(close.iloc[-1]) > float(ma10.iloc[-1])
        and float(ma10.iloc[-1]) > float(ma40.iloc[-1])
        and float(ma40.iloc[-1]) > float(ma40.iloc[-8])
    )


def prior_uptrend(df):
    closes = df["Close"].tail(40)
    low    = float(closes.min())
    high   = float(closes.max())
    return (high - low) / low > 0.20 if low > 0 else False


def detect_vcp(df):
    bars         = df.tail(10)
    ranges       = (bars["High"] - bars["Low"]).values
    contractions = sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i - 1])
    if contractions < 3:
        return False, contractions, None
    pivot = float(df["High"].tail(5).max())
    return True, contractions, pivot


def contraction_volume_pattern(df):
    ranges  = (df["High"] - df["Low"]).values[-5:-2]
    volumes = df["Volume"].values[-5:-2]
    if len(ranges) < 3:
        return False
    return (
        ranges[0]  > ranges[1]  > ranges[2]
        and volumes[0] > volumes[1] > volumes[2]
    )


def tight_area(df):
    closes    = df["Close"].tail(3)
    pct_range = (float(closes.max()) - float(closes.min())) / float(closes.mean())
    return pct_range < 0.03


def volume_dryup(df):
    recent   = float(df["Volume"].tail(3).mean())
    previous = float(df["Volume"].tail(12).head(9).mean())
    return recent < previous if previous > 0 else False


def is_breaking_out(df, pivot):
    latest  = df.iloc[-1]
    avg_vol = float(df["Volume"].tail(10).mean())
    return (
        float(latest["High"]) > pivot
        and float(latest["Volume"]) > avg_vol * 1.5
    )

# ─────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────

def score_setup(df, spy, contractions, pct_from_pivot):
    rs    = relative_strength(df, spy)
    score = (
        contractions           * 20
        + (1 - pct_from_pivot) * 40
        + rs                   * 20
        + (10 if volume_dryup(df) else 0)
        + (10 if tight_area(df)   else 0)
    )
    return round(score, 2)

# ─────────────────────────────────────────
#  HTML + JSON OUTPUT
# ─────────────────────────────────────────

def export_html(results, scanned):
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = ""

    for rank, r in enumerate(results, 1):
        ticker  = r["Ticker"]
        badge   = ' <span style="color:#00ff88;font-weight:bold">⚡ BREAKOUT</span>' if r.get("Breakout") else ""
        chart   = (
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

    log.info("Exported %d candidates to results.html + results.json", len(results))

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

        time.sleep(2)

        for symbol in batch:
            scanned += 1

            try:
                df = extract_ticker(raw, symbol, batch)
                if df is None:
                    continue
                df = df.dropna(subset=["Close", "Volume"])
            except Exception:
                continue

            if len(df) < MIN_WEEKS:
                continue
            if not liquidity_ok(df):
                continue
            if not stage2(df):
                continue
            if not prior_uptrend(df):
                continue

            is_vcp, contractions, pivot = detect_vcp(df)

            if not is_vcp:
                continue
            if not contraction_volume_pattern(df):
                continue

            close          = float(df["Close"].iloc[-1])
            pct_from_pivot = (pivot - close) / pivot

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
                log.info("  VCP forming  %s  score=%.1f  %.1f%% from pivot", symbol, score, pct_from_pivot * 100)

    # ── Telegram alerts ──────────────────
    if breakout_alerts:
        lines = ["🚨 VCP BREAKOUT ALERT\n"]
        for r in breakout_alerts:
            lines.append(f"• {r['Ticker']}  pivot={r['Pivot']}  score={r['Score']}  ({r['PctFromPivot']}% from pivot)")
        lines.append(f"\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        send_telegram("\n".join(lines))

    elif results:
        top   = sorted(results, key=lambda x: x["Score"], reverse=True)[:5]
        lines = [f"📊 Scan complete — {len(results)} VCP setups forming\n"]
        for r in top:
            lines.append(f"• {r['Ticker']}  score={r['Score']}  ({r['PctFromPivot']}% from pivot)")
        send_telegram("\n".join(lines))

    else:
        send_telegram("📊 Scan complete — no VCP setups found this pass.")

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
