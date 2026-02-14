import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import requests
import os
import smtplib
import logging
import xml.etree.ElementTree as ET
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from deep_translator import GoogleTranslator

# ===== ログ設定 =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===== 銘柄設定 =====

tickers = {
    "NVDA": "NVIDIA",
    "MU": "Micron",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "^GSPC": "S&P500",
    "^IXIC": "NASDAQ",
    "SOXX": "SOXX",
}

VIX_TICKER = "^VIX"

MACRO_TICKERS = {
    "^TNX": "米10年債利回り",
    "JPY=X": "USD/JPY",
    "GC=F": "金(Gold)",
}

negative_keywords = [
    "lawsuit", "crash", "downgrade", "fraud", "investigation",
    "recall", "bankruptcy", "layoff", "decline",
    "missed earnings", "regulation", "antitrust",
]

semiconductor_stocks = ["NVDA", "AMD", "AVGO", "MU"]
ai_large_stocks = ["MSFT", "AMZN", "GOOGL"]
INDIVIDUAL_TICKERS = semiconductor_stocks + ai_large_stocks
EARNINGS_WATCH_TICKERS = INDIVIDUAL_TICKERS

# 相関監視ペア: (銘柄A, 銘柄B/ベンチマーク, 説明)
CORRELATION_PAIRS = [
    ("NVDA", "SOXX", "NVDA vs 半導体セクター"),
    ("AMD", "SOXX", "AMD vs 半導体セクター"),
    ("NVDA", "AMD", "NVDA vs AMD (半導体ライバル)"),
    ("MSFT", "^GSPC", "MSFT vs S&P500"),
    ("AMZN", "^GSPC", "AMZN vs S&P500"),
    ("SOXX", "^IXIC", "半導体 vs NASDAQ"),
]

# ===== FOMC日程 =====
FOMC_DATES_2025 = [
    datetime.date(2025, 1, 29), datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7), datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30), datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29), datetime.date(2025, 12, 10),
]
FOMC_DATES_2026 = [
    datetime.date(2026, 1, 28), datetime.date(2026, 3, 18),
    datetime.date(2026, 5, 6), datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29), datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28), datetime.date(2026, 12, 16),
]
FOMC_DATES = FOMC_DATES_2025 + FOMC_DATES_2026

EARNINGS_WARN_DAYS = 7
EARNINGS_POST_DAYS = 2
FOMC_WARN_DAYS = 5


# ===== ユーティリティ =====


def translate_to_japanese(text: str) -> str:
    try:
        return GoogleTranslator(source="en", target="ja").translate(text)
    except Exception as e:
        logger.warning(f"翻訳失敗: {e}")
        return text


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) を計算"""
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(period).mean()


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def diff_arrow(current: float, previous: float) -> str:
    diff = current - previous
    if abs(diff) < 0.01:
        return "→ (変化なし)"
    arrow = "↑" if diff > 0 else "↓"
    return f"{arrow} {diff:+.2f}"


# ===== 決算・イベントカレンダー =====


def fetch_earnings_calendar() -> list[dict]:
    today = datetime.date.today()
    earnings = []
    for ticker in EARNINGS_WATCH_TICKERS:
        name = tickers.get(ticker, ticker)
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
                continue
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = ed[0]
                    elif isinstance(ed, (datetime.datetime, datetime.date)):
                        earnings_date = ed
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    vals = cal["Earnings Date"].dropna()
                    if len(vals) > 0:
                        earnings_date = vals.iloc[0]
                elif "Earnings Date" in cal.index:
                    vals = cal.loc["Earnings Date"].dropna()
                    if len(vals) > 0:
                        earnings_date = vals.iloc[0]
            if earnings_date is None:
                continue
            if isinstance(earnings_date, (datetime.datetime, pd.Timestamp)):
                earnings_date = earnings_date.date()
            days_until = (earnings_date - today).days
            earnings.append({
                "ticker": ticker, "name": name,
                "date": earnings_date, "days_until": days_until,
            })
        except Exception as e:
            logger.warning(f"{ticker} 決算日取得失敗: {e}")
    earnings.sort(key=lambda x: x["date"])
    return earnings


def get_upcoming_fomc() -> list[dict]:
    today = datetime.date.today()
    upcoming = []
    for d in FOMC_DATES:
        days_until = (d - today).days
        if days_until >= -1:
            upcoming.append({"date": d, "days_until": days_until})
        if len(upcoming) >= 3:
            break
    return upcoming


def build_event_alerts(earnings, fomc):
    alerts = []
    score_adj = 0
    for e in earnings:
        ticker, days = e["ticker"], e["days_until"]
        date_str = e["date"].strftime("%m/%d")
        if 0 < days <= EARNINGS_WARN_DAYS:
            urgency = "🔴" if days <= 3 else "🟡"
            alerts.append(f"{urgency} {e['name']} ({ticker}) 決算まで{days}日 ({date_str})")
            if ticker in ("NVDA", "AMD") and days <= 3:
                score_adj -= 3
                alerts.append(f"   → {ticker} 決算直前: ポジション縮小推奨")
        elif days == 0:
            alerts.append(f"🔔 {e['name']} ({ticker}) 本日決算発表！ ({date_str})")
            if ticker in ("NVDA", "AMD"):
                score_adj -= 5
                alerts.append(f"   → {ticker} 決算当日: 高ボラティリティに警戒")
        elif -EARNINGS_POST_DAYS <= days < 0:
            alerts.append(f"📋 {e['name']} ({ticker}) 決算発表済み ({date_str}) 結果注視")
    for f in fomc:
        days, date_str = f["days_until"], f["date"].strftime("%m/%d")
        if 0 < days <= FOMC_WARN_DAYS:
            urgency = "🔴" if days <= 2 else "🟡"
            alerts.append(f"{urgency} FOMC まで{days}日 ({date_str})")
            if days <= 2:
                score_adj -= 3
                alerts.append("   → FOMC直前: 様子見推奨")
        elif days == 0:
            alerts.append(f"🔔 本日FOMC発表！ ({date_str})")
            score_adj -= 5
            alerts.append("   → FOMC当日: 結果待ちでポジション縮小推奨")
        elif days == -1:
            alerts.append(f"📋 FOMC結果発表直後 ({date_str}) 市場反応を注視")
    return alerts, score_adj


# ===== テクニカルデータ取得（ATR・出来高プロファイル拡張） =====


# 年間ヒストリーのキャッシュ（銘柄ごとにAPIを1回だけ叩く）
_hist_cache: dict[str, pd.DataFrame] = {}


def get_history(ticker_symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """ヒストリーをキャッシュ付きで取得"""
    if ticker_symbol in _hist_cache:
        return _hist_cache[ticker_symbol]
    try:
        hist = yf.Ticker(ticker_symbol).history(period=period)
        if len(hist) >= 2:
            _hist_cache[ticker_symbol] = hist
            return hist
    except Exception as e:
        logger.error(f"{ticker_symbol} データ取得失敗: {e}")
    return None


def fetch_technical_data(ticker_symbol: str) -> dict | None:
    hist = get_history(ticker_symbol)
    if hist is None or len(hist) < 200:
        if hist is not None:
            logger.warning(f"{ticker_symbol}: データ不足 ({len(hist)}行)")
        return None

    close = hist["Close"]
    current = close.iloc[-1]
    prev = close.iloc[-2]

    prev2 = close.iloc[-3] if len(close) >= 3 else None
    prev_change = round((prev / prev2 - 1) * 100, 2) if prev2 else None

    rsi_series = calculate_rsi(close)
    rsi_today = round(rsi_series.iloc[-1], 2)
    rsi_prev = round(rsi_series.iloc[-2], 2) if len(rsi_series) >= 2 else None

    vol_avg = hist["Volume"].rolling(20).mean().iloc[-1]
    vol_current = hist["Volume"].iloc[-1]
    if pd.notna(vol_avg) and vol_avg > 0 and pd.notna(vol_current):
        volume_ratio = round(vol_current / vol_avg, 2)
    else:
        volume_ratio = None

    # ATR
    atr_series = calculate_atr(hist)
    atr_current = round(atr_series.iloc[-1], 2) if pd.notna(atr_series.iloc[-1]) else None
    atr_avg = round(atr_series.rolling(50).mean().iloc[-1], 2) if len(atr_series) >= 50 else None
    atr_ratio = round(atr_current / atr_avg, 2) if atr_current and atr_avg and atr_avg > 0 else None
    # ATR%（価格に対するATRの割合）
    atr_pct = round((atr_current / current) * 100, 2) if atr_current and current > 0 else None

    return {
        "current": round(current, 2),
        "change": round((current / prev - 1) * 100, 2),
        "prev_change": prev_change,
        "ma50": round(close.rolling(50).mean().iloc[-1], 2),
        "ma200": round(close.rolling(200).mean().iloc[-1], 2),
        "rsi": rsi_today,
        "rsi_prev": rsi_prev,
        "volume_ratio": volume_ratio,
        "atr": atr_current,
        "atr_avg": atr_avg,
        "atr_ratio": atr_ratio,
        "atr_pct": atr_pct,
    }


# ===== 出来高プロファイル解析 =====


def analyze_volume_profile(ticker_symbol: str, lookback: int = 60) -> dict | None:
    """
    直近N日の出来高プロファイルを分析。
    価格帯ごとの出来高分布からPOC（Point of Control）と
    Value Area（出来高70%集中帯）を算出。
    """
    hist = get_history(ticker_symbol)
    if hist is None or len(hist) < lookback:
        return None

    recent = hist.tail(lookback)
    close = recent["Close"]
    volume = recent["Volume"]

    if volume.sum() == 0 or pd.isna(volume.sum()):
        return None

    # 価格を20等分してビン化
    price_min = close.min()
    price_max = close.max()
    if price_max == price_min:
        return None

    n_bins = 20
    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 各ビンの出来高合計
    vol_profile = np.zeros(n_bins)
    for i in range(len(close)):
        price = close.iloc[i]
        vol = volume.iloc[i]
        if pd.isna(price) or pd.isna(vol):
            continue
        bin_idx = min(int((price - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
        vol_profile[bin_idx] += vol

    # POC（最大出来高の価格帯）
    poc_idx = np.argmax(vol_profile)
    poc_price = round(float(bin_centers[poc_idx]), 2)

    # Value Area（出来高70%が集中する範囲）
    total_vol = vol_profile.sum()
    if total_vol == 0:
        return None

    sorted_indices = np.argsort(vol_profile)[::-1]
    cumulative = 0.0
    va_bins = []
    for idx in sorted_indices:
        cumulative += vol_profile[idx]
        va_bins.append(idx)
        if cumulative >= total_vol * 0.7:
            break

    va_low = round(float(bins[min(va_bins)]), 2)
    va_high = round(float(bins[max(va_bins) + 1]), 2)

    current_price = round(float(close.iloc[-1]), 2)

    # 現在値がVA内かどうか
    if current_price < va_low:
        position = "VA下方"
    elif current_price > va_high:
        position = "VA上方"
    else:
        position = "VA内"

    return {
        "poc": poc_price,
        "va_low": va_low,
        "va_high": va_high,
        "current": current_price,
        "position": position,
        "poc_distance_pct": round((current_price / poc_price - 1) * 100, 2),
    }


# ===== 相関・乖離分析 =====


def analyze_correlations() -> list[dict]:
    """
    銘柄ペアの相関と直近の乖離を分析。
    60日相関と5日リターン乖離を比較。
    """
    results = []

    for ticker_a, ticker_b, label in CORRELATION_PAIRS:
        hist_a = get_history(ticker_a)
        hist_b = get_history(ticker_b)

        if hist_a is None or hist_b is None:
            continue
        if len(hist_a) < 60 or len(hist_b) < 60:
            continue

        # 日次リターン
        ret_a = hist_a["Close"].pct_change().dropna()
        ret_b = hist_b["Close"].pct_change().dropna()

        # 共通インデックス
        common = ret_a.index.intersection(ret_b.index)
        if len(common) < 60:
            continue

        ret_a = ret_a.loc[common]
        ret_b = ret_b.loc[common]

        # 60日相関
        corr_60d = round(ret_a.tail(60).corr(ret_b.tail(60)), 3)

        # 5日リターン
        ret_5d_a = round((hist_a["Close"].iloc[-1] / hist_a["Close"].iloc[-6] - 1) * 100, 2) if len(hist_a) >= 6 else None
        ret_5d_b = round((hist_b["Close"].iloc[-1] / hist_b["Close"].iloc[-6] - 1) * 100, 2) if len(hist_b) >= 6 else None

        if ret_5d_a is None or ret_5d_b is None:
            continue

        divergence = round(ret_5d_a - ret_5d_b, 2)

        # 乖離の大きさを判定
        is_diverging = abs(divergence) > 3.0 and corr_60d > 0.5

        results.append({
            "label": label,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "corr_60d": corr_60d,
            "ret_5d_a": ret_5d_a,
            "ret_5d_b": ret_5d_b,
            "divergence": divergence,
            "is_diverging": is_diverging,
        })

    return results


# ===== チャンス銘柄自動抽出 =====


def scan_opportunities(report_data: dict, volume_profiles: dict) -> list[dict]:
    """
    複数シグナルの複合判定でチャンス銘柄を抽出。
    各チャンスに「なぜ今チャンスなのか」の根拠を付与。
    """
    opportunities = []

    for ticker in INDIVIDUAL_TICKERS:
        d = report_data.get(ticker)
        if d is None:
            continue

        name = tickers.get(ticker, ticker)
        signals = []
        score = 0

        # --- シグナル1: RSI oversold bounce ---
        if d["rsi"] < 35:
            signals.append(f"RSI={d['rsi']}で売られすぎ水準")
            score += 2
            if d.get("rsi_prev") and d["rsi"] > d["rsi_prev"]:
                signals.append("RSI反転上昇の兆し")
                score += 1

        # --- シグナル2: 上昇トレンド内の押し目 ---
        if d["ma50"] > d["ma200"] and d["change"] < -2:
            signals.append(f"上昇トレンド中（MA50>MA200）に{d['change']}%の下落 → 押し目の可能性")
            score += 2

        # --- シグナル3: ATRボラティリティ収縮 → ブレイクアウト前兆 ---
        if d.get("atr_ratio") and d["atr_ratio"] < 0.7:
            signals.append(f"ATR比率={d['atr_ratio']}（平均の{int(d['atr_ratio']*100)}%）ボラ収縮 → ブレイクアウト前兆")
            score += 2

        # --- シグナル4: 出来高急増 + 上昇 ---
        if d.get("volume_ratio") and d["volume_ratio"] > 1.5 and d["change"] > 0:
            signals.append(f"出来高{d['volume_ratio']}倍に急増 + 上昇 → 買い集め示唆")
            score += 2

        # --- シグナル5: POC（出来高集中帯）での反発 ---
        vp = volume_profiles.get(ticker)
        if vp:
            if vp["position"] == "VA下方" and d["change"] > 0:
                signals.append(f"VA下方({vp['va_low']})から反発中 → 出来高サポートゾーン")
                score += 1
            elif abs(vp["poc_distance_pct"]) < 1.5:
                signals.append(f"POC({vp['poc']})付近 → 出来高集中価格帯でサポート")
                score += 1

        # --- シグナル6: ATR拡大 + 強い上昇 ---
        if d.get("atr_ratio") and d["atr_ratio"] > 1.5 and d["change"] > 3:
            signals.append(f"ATR拡大({d['atr_ratio']}倍) + 大幅上昇{d['change']}% → モメンタム発生")
            score += 2

        # --- シグナル7: MA50へのプルバック ---
        if d["ma50"] > d["ma200"] and d["current"] > 0:
            ma50_dist = abs((d["current"] / d["ma50"] - 1) * 100)
            if ma50_dist < 1.5 and d["rsi"] < 50:
                signals.append(f"MA50({d['ma50']})付近まで調整 → サポートで反発期待")
                score += 2

        # 閾値: 2シグナル以上でチャンス認定
        if score >= 3 and len(signals) >= 2:
            # 判定（攻めか守りか）
            if d["change"] < -2 and d["rsi"] < 40:
                action = "🔵 押し目買い候補"
            elif d.get("atr_ratio") and d["atr_ratio"] < 0.7:
                action = "🟡 ブレイクアウト待ち"
            elif d["change"] > 2 and d.get("volume_ratio") and d["volume_ratio"] > 1.5:
                action = "🟢 モメンタム乗り"
            else:
                action = "🔵 注目"

            opportunities.append({
                "ticker": ticker,
                "name": name,
                "action": action,
                "score": score,
                "signals": signals,
                "current": d["current"],
                "change": d["change"],
                "rsi": d["rsi"],
            })

    # スコア順にソート
    opportunities.sort(key=lambda x: x["score"], reverse=True)
    return opportunities


# ===== マクロ指標取得 =====


def fetch_macro_data() -> dict:
    macro = {}
    for ticker, name in MACRO_TICKERS.items():
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if len(data) >= 2:
                current = round(data["Close"].iloc[-1], 2)
                prev = round(data["Close"].iloc[-2], 2)
                change = round((current / prev - 1) * 100, 2)
                macro[ticker] = {"name": name, "current": current, "prev": prev, "change": change}
        except Exception as e:
            logger.error(f"{ticker} マクロデータ取得失敗: {e}")
    return macro


# ===== 週次サマリー =====


def fetch_weekly_performance() -> dict:
    weekly = {}
    all_watch = list(tickers.keys()) + [VIX_TICKER]
    for ticker in all_watch:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                weekly[ticker] = {
                    "open": round(hist["Close"].iloc[0], 2),
                    "close": round(hist["Close"].iloc[-1], 2),
                    "high": round(hist["High"].max(), 2),
                    "low": round(hist["Low"].min(), 2),
                    "change": round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2),
                }
        except Exception as e:
            logger.warning(f"{ticker} 週次データ取得失敗: {e}")
    return weekly


def generate_weekly_summary(weekly, score, temp, earnings_calendar):
    lines = []
    lines.append("")
    lines.append("=" * 40)
    lines.append("📈 週次サマリー")
    lines.append("=" * 40)

    lines.append("")
    lines.append("■ 週間パフォーマンス")
    lines.append("-" * 30)

    stock_tickers = [t for t in weekly if not t.startswith("^") and t != "SOXX"]
    index_tickers = [t for t in weekly if t.startswith("^") or t == "SOXX"]

    sorted_stocks = sorted(stock_tickers, key=lambda t: weekly[t]["change"], reverse=True)

    lines.append("  [個別銘柄]")
    for rank, ticker in enumerate(sorted_stocks, 1):
        w = weekly[ticker]
        name = tickers.get(ticker, ticker)
        sign = "+" if w["change"] >= 0 else ""
        bar_len = min(abs(int(w["change"])), 20)
        bar = "🟩" * bar_len if w["change"] >= 0 else "🟥" * bar_len
        lines.append(f"  {rank}. {name:10s} {sign}{w['change']:6.2f}%  {bar}")
        lines.append(f"     始値: {w['open']}  終値: {w['close']}  高値: {w['high']}  安値: {w['low']}")

    lines.append("")
    lines.append("  [指数・ETF]")
    for ticker in index_tickers:
        if ticker not in weekly:
            continue
        w = weekly[ticker]
        name = tickers.get(ticker, "VIX" if ticker == VIX_TICKER else ticker)
        sign = "+" if w["change"] >= 0 else ""
        lines.append(f"  {name:10s} {sign}{w['change']:6.2f}%  ({w['open']} → {w['close']})")

    if sorted_stocks:
        best, worst = sorted_stocks[0], sorted_stocks[-1]
        lines.append("")
        lines.append("■ 週間ハイライト")
        lines.append("-" * 30)
        lines.append(f"  🏆 ベスト:  {tickers.get(best, best)} ({best})  {weekly[best]['change']:+.2f}%")
        lines.append(f"  📉 ワースト: {tickers.get(worst, worst)} ({worst})  {weekly[worst]['change']:+.2f}%")

    lines.append("")
    lines.append("■ 来週の注目イベント")
    lines.append("-" * 30)
    today = datetime.date.today()
    next_events = []
    for e in earnings_calendar:
        if 1 <= e["days_until"] <= 14:
            next_events.append(f"  📅 {e['name']} ({e['ticker']}) 決算  {e['date'].strftime('%m/%d (%a)')}  ({e['days_until']}日後)")
    for d in FOMC_DATES:
        days_until = (d - today).days
        if 1 <= days_until <= 14:
            next_events.append(f"  🏛 FOMC  {d.strftime('%m/%d (%a)')}  ({days_until}日後)")
    if next_events:
        for ev in next_events:
            lines.append(ev)
    else:
        lines.append("  特になし")

    lines.append("")
    lines.append("■ 週間総評")
    lines.append("-" * 30)
    if stock_tickers:
        avg_change = np.mean([weekly[t]["change"] for t in stock_tickers])
        positive_count = sum(1 for t in stock_tickers if weekly[t]["change"] > 0)
        total = len(stock_tickers)
        if avg_change > 2:
            verdict = "📈 強い上昇の1週間。モメンタムの持続性に注目。"
        elif avg_change > 0:
            verdict = "➡ 小幅上昇。方向感の確認が必要。"
        elif avg_change > -2:
            verdict = "➡ 小幅下落。押し目形成の可能性も。"
        else:
            verdict = "📉 大幅下落の1週間。リスク管理の徹底を。"
        lines.append(f"  監視銘柄平均: {avg_change:+.2f}%  ({positive_count}/{total}銘柄が上昇)")
        lines.append(f"  {verdict}")
        lines.append(f"  来週の市場温度見通し: {score} {temp}")

    return "\n".join(lines)


# ===== スコアリング =====


def calculate_score(report_data, vix_data, macro_data):
    score = 50
    risk_flag = False

    for ticker, d in report_data.items():
        change = d["change"]
        if ticker in ("NVDA", "AMD"):
            if change > 2: score += 5
            if d["current"] > d["ma50"]: score += 5
            if d["rsi"] < 30: score += 3
            if d["rsi"] > 70: score -= 3
        if ticker == "^GSPC":
            if change > 1: score += 5
            if change < -1: score -= 5

    soxx = report_data.get("SOXX")
    if soxx and soxx["current"] < soxx["ma200"]:
        score -= 15
        risk_flag = True

    nasdaq = report_data.get("^IXIC")
    if nasdaq and nasdaq["current"] < nasdaq["ma200"]:
        score -= 10
        risk_flag = True

    if vix_data:
        vix_change = vix_data["change"]
        if vix_change > 10:
            score -= 15; risk_flag = True
        elif vix_change > 5:
            score -= 10; risk_flag = True

    tnx = macro_data.get("^TNX")
    if tnx:
        if tnx["change"] > 3: score -= 5
        elif tnx["change"] < -3: score += 3

    usdjpy = macro_data.get("JPY=X")
    if usdjpy and usdjpy["change"] < -1.5:
        score -= 3

    gold = macro_data.get("GC=F")
    if gold and gold["change"] > 2:
        score -= 3

    return int(clamp(score)), risk_flag


def get_temperature_label(score):
    if score >= 80: return "🔥 加速局面"
    elif score >= 65: return "🟢 強気"
    elif score >= 45: return "⚖ 中立"
    elif score >= 30: return "🟡 減速"
    else: return "❄ 崩れ"


# ===== ポジション配分 =====


def get_allocation(score, risk_flag):
    if risk_flag:
        return {"cash": 70, "semiconductor": 5, "ai_large": 5, "defensive": 20}
    if score >= 80:
        return {"cash": 10, "semiconductor": 40, "ai_large": 40, "defensive": 10}
    elif score >= 65:
        return {"cash": 20, "semiconductor": 35, "ai_large": 35, "defensive": 10}
    elif score >= 45:
        return {"cash": 35, "semiconductor": 25, "ai_large": 25, "defensive": 15}
    elif score >= 30:
        return {"cash": 50, "semiconductor": 15, "ai_large": 15, "defensive": 20}
    else:
        return {"cash": 70, "semiconductor": 5, "ai_large": 5, "defensive": 20}


def distribute(group, total_weight, report_data):
    strong, normal, reduced = [], [], []
    for ticker in group:
        d = report_data.get(ticker)
        if d is None: continue
        trend_ok = d["ma50"] > d["ma200"]
        rsi = d["rsi"]
        if d["change"] > 2 and trend_ok and 40 <= rsi <= 65:
            strong.append(ticker)
        elif rsi > 70:
            reduced.append(ticker)
        else:
            normal.append(ticker)
    active = [t for t in group if t in report_data]
    if not active: return {}
    if not strong and not normal and not reduced:
        per = round(total_weight / len(active), 1)
        return {t: per for t in active}
    if strong:
        weights = {"strong": 0.6, "normal": 0.3, "reduced": 0.1}
    else:
        weights = {"strong": 0.0, "normal": 0.8, "reduced": 0.2}
    result = {}
    for bucket, tl in [("strong", strong), ("normal", normal), ("reduced", reduced)]:
        if tl:
            per = round(total_weight * weights[bucket] / len(tl), 1)
            for t in tl: result[t] = per
    return result


def build_detailed_allocation(allocation, report_data):
    detailed = {}
    detailed.update(distribute(semiconductor_stocks, allocation["semiconductor"], report_data))
    detailed.update(distribute(ai_large_stocks, allocation["ai_large"], report_data))
    return detailed


def apply_nvda_boost(detailed, score, risk_flag, report_data):
    if "NVDA" not in detailed: return detailed
    boost = 0
    if score >= 65: boost += 5
    if risk_flag: boost -= 5
    if "NVDA" in report_data and report_data["NVDA"]["rsi"] < 35: boost += 3
    detailed["NVDA"] = max(0, detailed["NVDA"] + boost)
    return detailed


def apply_vix_adjustment(detailed, vix_data, allocation):
    vix_change = vix_data["change"] if vix_data else 0
    vol_factor = 1.0
    if vix_change > 5: vol_factor = 0.8
    elif vix_change < -3: vol_factor = 1.1
    for t in detailed:
        detailed[t] = round(detailed[t] * vol_factor, 1)
    total_weight = sum(detailed.values())
    target_total = allocation["semiconductor"] + allocation["ai_large"]
    if total_weight > 0:
        scale = target_total / total_weight
        for t in detailed:
            detailed[t] = round(detailed[t] * scale, 1)
    return detailed


# ===== ニュース取得 =====


def get_ai_news_from_newsapi():
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key: return []
    from_date = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
    url = (
        "https://newsapi.org/v2/everything?"
        "q=(AI OR semiconductor) AND (NVIDIA OR AMD OR Micron OR Broadcom)&"
        f"from={from_date}&language=en&sortBy=publishedAt&pageSize=5&"
        f"apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok": return []
        return [a["title"] for a in data.get("articles", []) if a.get("title") and a["title"] != "[Removed]"][:5]
    except Exception as e:
        logger.warning(f"NewsAPI取得失敗: {e}")
        return []


def get_ai_news_from_rss():
    queries = ["AI semiconductor", "NVIDIA AMD stock"]
    titles = []
    for query in queries:
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:3]:
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    titles.append(title_el.text)
        except Exception as e:
            logger.error(f"RSS取得失敗 ({query}): {e}")
    seen = set()
    return [t for t in titles if not (t in seen or seen.add(t))][:5]


def get_ai_news():
    news = get_ai_news_from_newsapi()
    if news:
        logger.info(f"NewsAPIから{len(news)}件取得")
        return news
    logger.info("NewsAPI失敗、RSSフォールバック")
    return get_ai_news_from_rss()


def analyze_news(news):
    translated, negative_count = [], 0
    for title in news:
        lower = title.lower()
        for word in negative_keywords:
            if word in lower: negative_count += 1
        translated.append(translate_to_japanese(title))
    return translated, negative_count


def is_rebalance_day():
    return datetime.date.today().weekday() == 4


# ===== レポート生成 =====


def generate_report(
    score, temp, risk_flag, report_data, vix_data, macro_data,
    translated_news, negative_count, allocation, detailed_allocation,
    rebalance, event_alerts, earnings_calendar, fomc_upcoming,
    opportunities, correlations, volume_profiles,
):
    lines = []

    if rebalance:
        lines.append("🔁 今週はリバランス実行日です")
    else:
        lines.append("📌 今週は配分維持日です")

    lines.append("")
    lines.append("=" * 40)
    lines.append(f"📊 AI市場プロレポート | {datetime.date.today()}")
    lines.append(f"市場温度: {score} {temp}")
    lines.append("=" * 40)

    # イベントアラート
    if event_alerts:
        lines.append("")
        lines.append("■ イベントアラート")
        lines.append("-" * 30)
        for alert in event_alerts:
            lines.append(f"  {alert}")

    # ========== チャンス銘柄（新セクション） ==========
    lines.append("")
    lines.append("■ チャンス銘柄スキャナー")
    lines.append("-" * 30)
    if opportunities:
        for opp in opportunities:
            lines.append(f"  {opp['action']} {opp['name']} ({opp['ticker']})")
            lines.append(f"    現在値: {opp['current']}  前日比: {opp['change']:+.2f}%  RSI: {opp['rsi']}  スコア: {opp['score']}")
            lines.append(f"    [根拠]")
            for sig in opp["signals"]:
                lines.append(f"      • {sig}")
            lines.append("")
    else:
        lines.append("  現在、明確なチャンスシグナルなし")

    # ========== 相関乖離アラート（新セクション） ==========
    lines.append("")
    lines.append("■ 相関・乖離モニター")
    lines.append("-" * 30)
    diverging = [c for c in correlations if c["is_diverging"]]
    if diverging:
        for c in diverging:
            lines.append(f"  ⚠ {c['label']}")
            lines.append(f"    60日相関: {c['corr_60d']}  5日リターン乖離: {c['divergence']:+.2f}%")
            lines.append(f"    {c['ticker_a']}: {c['ret_5d_a']:+.2f}%  {c['ticker_b']}: {c['ret_5d_b']:+.2f}%")
            # 乖離の解釈
            if c["divergence"] > 0:
                lines.append(f"    → {c['ticker_a']}が相対的に強い（{c['ticker_b']}へのキャッチアップ or 独自材料の可能性）")
            else:
                lines.append(f"    → {c['ticker_a']}が相対的に弱い（修正リスク or 固有の悪材料の可能性）")
            lines.append("")
    else:
        lines.append("  主要ペアに大きな乖離なし（正常範囲）")

    # 全ペアの相関一覧（コンパクト表示）
    lines.append("")
    lines.append("  [相関一覧]")
    for c in correlations:
        div_mark = " ⚠" if c["is_diverging"] else ""
        lines.append(f"    {c['label']:25s}  相関:{c['corr_60d']:+.3f}  乖離:{c['divergence']:+.2f}%{div_mark}")

    # 決算カレンダー
    lines.append("")
    lines.append("■ 決算カレンダー（今後30日）")
    lines.append("-" * 30)
    upcoming = [e for e in earnings_calendar if 0 <= e["days_until"] <= 30]
    if upcoming:
        for e in upcoming:
            date_str = e["date"].strftime("%m/%d (%a)")
            days = e["days_until"]
            tag = "⚡本日" if days == 0 else f"🔴 {days}日後" if days <= 3 else f"🟡 {days}日後" if days <= 7 else f"   {days}日後"
            lines.append(f"  {tag}  {e['name']} ({e['ticker']})  {date_str}")
    else:
        lines.append("  今後30日以内の決算予定なし")

    # FOMC
    lines.append("")
    lines.append("■ FOMC日程")
    lines.append("-" * 30)
    if fomc_upcoming:
        for f in fomc_upcoming:
            date_str = f["date"].strftime("%m/%d (%a)")
            days = f["days_until"]
            tag = "⚡本日" if days == 0 else f"🔴 {days}日後" if days <= 3 else f"🟡 {days}日後" if days <= 7 else f"   {days}日後"
            lines.append(f"  {tag}  FOMC  {date_str}")
    else:
        lines.append("  直近のFOMC日程なし")

    # マクロ環境
    lines.append("")
    lines.append("■ マクロ環境")
    lines.append("-" * 30)
    if macro_data:
        for ticker, m in macro_data.items():
            sign = "+" if m["change"] >= 0 else ""
            val = f"{m['current']}%" if ticker == "^TNX" else f"{m['current']}"
            lines.append(f"  {m['name']}: {val}  ({sign}{m['change']}%)  {diff_arrow(m['current'], m['prev'])}")
        warnings = []
        tnx = macro_data.get("^TNX")
        usdjpy = macro_data.get("JPY=X")
        gold = macro_data.get("GC=F")
        if tnx and tnx["change"] > 3: warnings.append("⚠ 金利急騰 → グロース株に逆風")
        if tnx and tnx["change"] < -3: warnings.append("✅ 金利低下 → グロース株に追い風")
        if usdjpy and usdjpy["change"] < -1.5: warnings.append("⚠ 急速な円高 → ドル建て資産目減り注意")
        if usdjpy and usdjpy["change"] > 1.5: warnings.append("✅ 円安進行 → ドル建て資産に追い風")
        if gold and gold["change"] > 2: warnings.append("⚠ 金価格急騰 → リスクオフの兆候")
        if warnings:
            lines.append("")
            for w in warnings: lines.append(f"  {w}")
    else:
        lines.append("  データ取得なし")

    # 銘柄テクニカル（ATR・VP付き）
    lines.append("")
    lines.append("■ 銘柄テクニカル")
    lines.append("-" * 30)

    all_tickers = {**tickers, VIX_TICKER: "VIX"}
    for ticker, name in all_tickers.items():
        d = report_data.get(ticker) if ticker != VIX_TICKER else vix_data
        if d is None: continue

        earnings_mark = ""
        for e in earnings_calendar:
            if e["ticker"] == ticker and 0 <= e["days_until"] <= EARNINGS_WARN_DAYS:
                earnings_mark = " ⚡決算本日" if e["days_until"] == 0 else f" 📅決算{e['days_until']}日後"
                break

        lines.append(f"  {name} ({ticker}){earnings_mark}")

        change_str = f"{d['change']:+.2f}%"
        if d.get("prev_change") is not None:
            momentum = d["change"] - d["prev_change"]
            m_icon = "📈 加速" if momentum > 0.5 else "📉 減速" if momentum < -0.5 else "➡ 横ばい"
            lines.append(f"    前日比: {change_str}  (前日: {d['prev_change']:+.2f}%) {m_icon}")
        else:
            lines.append(f"    前日比: {change_str}")

        lines.append(f"    MA50: {d['ma50']}  MA200: {d['ma200']}")

        rsi_str = f"{d['rsi']}"
        if d.get("rsi_prev") is not None:
            rsi_str += f"  ({diff_arrow(d['rsi'], d['rsi_prev'])})"
        lines.append(f"    RSI: {rsi_str}")

        vol_str = f"{d['volume_ratio']}倍" if d["volume_ratio"] is not None else "N/A"
        lines.append(f"    出来高倍率: {vol_str}")

        # ATR表示
        if d.get("atr") is not None:
            atr_line = f"    ATR: {d['atr']}"
            if d.get("atr_pct") is not None:
                atr_line += f"  ({d['atr_pct']}%)"
            if d.get("atr_ratio") is not None:
                if d["atr_ratio"] > 1.3:
                    atr_line += f"  ⚠ 平均の{int(d['atr_ratio']*100)}%（高ボラ）"
                elif d["atr_ratio"] < 0.7:
                    atr_line += f"  📊 平均の{int(d['atr_ratio']*100)}%（低ボラ）"
                else:
                    atr_line += f"  平均の{int(d['atr_ratio']*100)}%"
            lines.append(atr_line)

        # 出来高プロファイル表示
        vp = volume_profiles.get(ticker)
        if vp:
            lines.append(f"    VP: POC={vp['poc']}  VA=[{vp['va_low']}-{vp['va_high']}]  位置: {vp['position']}")

        lines.append("")

    # ニュース
    lines.append("■ AI関連最新ニュース")
    lines.append("-" * 30)
    if translated_news:
        for n in translated_news: lines.append(f"  - {n}")
    else:
        lines.append("  - ニュースの取得なし")
    if negative_count >= 2:
        lines.append("")
        lines.append("  ⚠ ネガティブニュース増加（市場警戒）")
    if risk_flag:
        lines.append("")
        lines.append("  ⚠ 崩れモード発動")

    # 押し目候補
    lines.append("")
    lines.append("■ 押し目候補")
    lines.append("-" * 30)
    dip_found = False
    for ticker, d in report_data.items():
        if d["change"] < -4 and d["rsi"] < 35 and d["ma50"] > d["ma200"]:
            lines.append(f"  ✅ {ticker} (RSI: {d['rsi']}, 前日比: {d['change']}%)")
            dip_found = True
    if not dip_found:
        lines.append("  該当なし")

    # ポジション配分
    lines.append("")
    lines.append("■ 推奨ポジション配分")
    lines.append("-" * 30)
    lines.append(f"  現金:          {allocation['cash']}%")
    lines.append(f"  半導体:        {allocation['semiconductor']}%")
    lines.append(f"  AI大型株:      {allocation['ai_large']}%")
    lines.append(f"  ディフェンシブ: {allocation['defensive']}%")

    lines.append("")
    lines.append("■ 銘柄別詳細配分")
    lines.append("-" * 30)
    for t, w in detailed_allocation.items():
        bar = "█" * int(w / 2)
        lines.append(f"  {t:6s}: {w:5.1f}%  {bar}")

    return "\n".join(lines)


# ===== メール送信 =====


def send_email(report, score, temp, risk_flag, macro_data, event_alerts, weekly_summary, opportunities):
    gmail_user = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_password:
        logger.error("Gmail認証情報が設定されていません")
        return

    # 件名
    if opportunities:
        subject = f"🎯 チャンス{len(opportunities)}件 + AI市場レポート"
    elif event_alerts and any("決算まで" in a and "🔴" in a for a in event_alerts):
        subject = "📅⚠ 決算接近アラート + AI市場レポート"
    elif risk_flag:
        subject = "⚠ AI市場警戒アラート"
    elif weekly_summary:
        subject = "📈 週次サマリー + AI市場レポート"
    else:
        subject = "📊 Daily AI Market Report"

    full_report = report
    if weekly_summary:
        full_report += "\n" + weekly_summary

    # マクロHTML
    macro_rows = ""
    for ticker, m in macro_data.items():
        sign = "+" if m["change"] >= 0 else ""
        color = "#e74c3c" if m["change"] < 0 else "#27ae60"
        val = f"{m['current']}%" if ticker == "^TNX" else f"{m['current']}"
        macro_rows += f'<tr><td style="padding:6px;">{m["name"]}</td><td style="padding:6px;">{val}</td><td style="padding:6px;color:{color};">{sign}{m["change"]}%</td></tr>'

    # イベントアラートHTML
    event_html = ""
    if event_alerts:
        items = "".join(f'<li style="margin:4px 0;">{a}</li>' for a in event_alerts)
        event_html = f'<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:12px;margin:15px 0;"><h3 style="margin:0 0 8px 0;">📅 イベントアラート</h3><ul style="margin:0;padding-left:20px;">{items}</ul></div>'

    # チャンス銘柄HTML
    opp_html = ""
    if opportunities:
        opp_items = ""
        for opp in opportunities:
            sigs = "<br>".join(f"• {s}" for s in opp["signals"])
            opp_items += f'<div style="margin:8px 0;padding:8px;background:#fff;border-radius:4px;"><b>{opp["action"]} {opp["name"]} ({opp["ticker"]})</b><br>現在値: {opp["current"]}  前日比: {opp["change"]:+.2f}%  RSI: {opp["rsi"]}<br><small style="color:#555;">{sigs}</small></div>'
        opp_html = f'<div style="background:#e3f2fd;border:1px solid #2196f3;border-radius:8px;padding:12px;margin:15px 0;"><h3 style="margin:0 0 8px 0;">🎯 チャンス銘柄</h3>{opp_items}</div>'

    weekly_html = ""
    if weekly_summary:
        weekly_html = f'<div style="background:#e8f5e9;border:1px solid #4caf50;border-radius:8px;padding:12px;margin:15px 0;"><h3 style="margin:0 0 8px 0;">📈 週次サマリー</h3><pre style="font-size:12px;line-height:1.5;margin:0;white-space:pre-wrap;">{weekly_summary}</pre></div>'

    score_color = "#27ae60" if score >= 65 else "#f39c12" if score >= 45 else "#e74c3c"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">
      <h2 style="border-bottom:2px solid #333;">📊 AI市場プロレポート</h2>
      <p><b>日付:</b> {datetime.date.today()}</p>
      <div style="margin:15px 0;">
        <span style="font-size:18px;font-weight:bold;">市場温度: {score} {temp}</span>
        <div style="background:#eee;border-radius:10px;height:20px;width:100%;margin-top:5px;">
          <div style="background:{score_color};height:20px;border-radius:10px;width:{score}%;"></div>
        </div>
      </div>
      {opp_html}
      {event_html}
      <h3>🌍 マクロ環境</h3>
      <table style="border-collapse:collapse;width:100%;">
        <tr style="background:#f5f5f5;"><th style="padding:8px;text-align:left;">指標</th><th style="padding:8px;text-align:left;">現在値</th><th style="padding:8px;text-align:left;">前日比</th></tr>
        {macro_rows}
      </table>
      {weekly_html}
      <hr style="margin:20px 0;">
      <pre style="font-size:13px;line-height:1.6;background:#f9f9f9;padding:15px;border-radius:8px;">{full_report}</pre>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = gmail_user
    msg.attach(MIMEText(full_report, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        logger.info("メール送信成功")
    except Exception as e:
        logger.error(f"メール送信失敗: {e}")


# ===== メイン処理 =====


def main():
    logger.info("===== AI市場監視Bot v4 起動 =====")

    # 1. テクニカルデータ取得（ATR含む）
    report_data = {}
    for ticker in tickers:
        data = fetch_technical_data(ticker)
        if data:
            report_data[ticker] = data

    vix_data = fetch_technical_data(VIX_TICKER)

    # 2. 出来高プロファイル解析
    volume_profiles = {}
    for ticker in INDIVIDUAL_TICKERS:
        vp = analyze_volume_profile(ticker)
        if vp:
            volume_profiles[ticker] = vp

    # 3. マクロ指標取得
    macro_data = fetch_macro_data()

    # 4. 相関・乖離分析
    correlations = analyze_correlations()

    # 5. 決算・イベントカレンダー取得
    earnings_calendar = fetch_earnings_calendar()
    fomc_upcoming = get_upcoming_fomc()
    event_alerts, event_score_adj = build_event_alerts(earnings_calendar, fomc_upcoming)

    # 6. スコアリング
    score, risk_flag = calculate_score(report_data, vix_data, macro_data)
    score = int(clamp(score + event_score_adj))

    # 7. ニュース分析
    news = get_ai_news()
    translated_news, negative_count = analyze_news(news)
    if negative_count >= 2:
        score = int(clamp(score - 10)); risk_flag = True
    elif negative_count == 1:
        score = int(clamp(score - 5))

    # 8. チャンス銘柄スキャン
    opportunities = scan_opportunities(report_data, volume_profiles)

    # 9. 温度判定
    temp = get_temperature_label(score)

    # 10. ポジション配分
    allocation = get_allocation(score, risk_flag)
    rebalance = is_rebalance_day()
    detailed_allocation = build_detailed_allocation(allocation, report_data)
    detailed_allocation = apply_nvda_boost(detailed_allocation, score, risk_flag, report_data)
    detailed_allocation = apply_vix_adjustment(detailed_allocation, vix_data, allocation)

    # 11. レポート生成
    report = generate_report(
        score, temp, risk_flag, report_data, vix_data, macro_data,
        translated_news, negative_count, allocation, detailed_allocation,
        rebalance, event_alerts, earnings_calendar, fomc_upcoming,
        opportunities, correlations, volume_profiles,
    )

    # 12. 週次サマリー（金曜のみ）
    weekly_summary = None
    if rebalance:
        logger.info("金曜日: 週次サマリー生成中...")
        weekly_summary = generate_weekly_summary(
            fetch_weekly_performance(), score, temp, earnings_calendar
        )

    logger.info("\n" + report)
    if weekly_summary:
        logger.info("\n" + weekly_summary)

    # 13. メール送信
    send_email(report, score, temp, risk_flag, macro_data, event_alerts, weekly_summary, opportunities)

    logger.info("===== 処理完了 =====")


if __name__ == "__main__":
    main()
