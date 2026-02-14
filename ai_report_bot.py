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

# VIXは別管理（二重取得・二重判定を防止）
VIX_TICKER = "^VIX"

# マクロ指標
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

# 決算を監視する個別銘柄（インデックス・ETFは除外）
EARNINGS_WATCH_TICKERS = ["NVDA", "MU", "AMD", "AVGO", "MSFT", "AMZN", "GOOGL"]

# ===== FOMC日程 =====
FOMC_DATES_2025 = [
    datetime.date(2025, 1, 29),
    datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7),
    datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30),
    datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29),
    datetime.date(2025, 12, 10),
]
FOMC_DATES_2026 = [
    datetime.date(2026, 1, 28),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 5, 6),
    datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28),
    datetime.date(2026, 12, 16),
]
FOMC_DATES = FOMC_DATES_2025 + FOMC_DATES_2026

EARNINGS_WARN_DAYS = 7
EARNINGS_POST_DAYS = 2
FOMC_WARN_DAYS = 5

# ===== ユーティリティ =====


def translate_to_japanese(text: str) -> str:
    """Google翻訳で日本語に変換（deep-translator使用）"""
    try:
        return GoogleTranslator(source="en", target="ja").translate(text)
    except Exception as e:
        logger.warning(f"翻訳失敗: {e}")
        return text


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相対力指数）を計算"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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
                "ticker": ticker,
                "name": name,
                "date": earnings_date,
                "days_until": days_until,
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


def build_event_alerts(
    earnings: list[dict], fomc: list[dict]
) -> tuple[list[str], int]:
    alerts = []
    score_adj = 0

    for e in earnings:
        ticker = e["ticker"]
        days = e["days_until"]
        date_str = e["date"].strftime("%m/%d")

        if 0 < days <= EARNINGS_WARN_DAYS:
            urgency = "🔴" if days <= 3 else "🟡"
            alerts.append(
                f"{urgency} {e['name']} ({ticker}) 決算まで{days}日 ({date_str})"
            )
            if ticker in ("NVDA", "AMD") and days <= 3:
                score_adj -= 3
                alerts.append(f"   → {ticker} 決算直前: ポジション縮小推奨")
        elif days == 0:
            alerts.append(
                f"🔔 {e['name']} ({ticker}) 本日決算発表！ ({date_str})"
            )
            if ticker in ("NVDA", "AMD"):
                score_adj -= 5
                alerts.append(f"   → {ticker} 決算当日: 高ボラティリティに警戒")
        elif -EARNINGS_POST_DAYS <= days < 0:
            alerts.append(
                f"📋 {e['name']} ({ticker}) 決算発表済み ({date_str}) 結果注視"
            )

    for f in fomc:
        days = f["days_until"]
        date_str = f["date"].strftime("%m/%d")

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


# ===== テクニカルデータ取得 =====


def fetch_technical_data(ticker_symbol: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
    except Exception as e:
        logger.error(f"{ticker_symbol} データ取得失敗: {e}")
        return None

    if len(hist) < 200:
        logger.warning(f"{ticker_symbol}: データ不足 ({len(hist)}行 < 200)")
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

    return {
        "current": round(current, 2),
        "change": round((current / prev - 1) * 100, 2),
        "prev_change": prev_change,
        "ma50": round(close.rolling(50).mean().iloc[-1], 2),
        "ma200": round(close.rolling(200).mean().iloc[-1], 2),
        "rsi": rsi_today,
        "rsi_prev": rsi_prev,
        "volume_ratio": volume_ratio,
    }


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
                macro[ticker] = {
                    "name": name,
                    "current": current,
                    "prev": prev,
                    "change": change,
                }
        except Exception as e:
            logger.error(f"{ticker} マクロデータ取得失敗: {e}")
    return macro


# ===== 週次サマリー =====


def fetch_weekly_performance() -> dict:
    """
    各銘柄の週間パフォーマンスを取得。
    5営業日分のデータから週初→週末の変化率を計算。
    """
    weekly = {}

    all_watch = list(tickers.keys()) + [VIX_TICKER]

    for ticker in all_watch:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                week_open = hist["Close"].iloc[0]
                week_close = hist["Close"].iloc[-1]
                week_high = hist["High"].max()
                week_low = hist["Low"].min()
                week_change = round((week_close / week_open - 1) * 100, 2)
                weekly[ticker] = {
                    "open": round(week_open, 2),
                    "close": round(week_close, 2),
                    "high": round(week_high, 2),
                    "low": round(week_low, 2),
                    "change": week_change,
                }
        except Exception as e:
            logger.warning(f"{ticker} 週次データ取得失敗: {e}")

    return weekly


def generate_weekly_summary(
    weekly: dict, score: int, temp: str, earnings_calendar: list[dict]
) -> str:
    """金曜日に追加される週次サマリーセクションを生成"""
    lines = []
    lines.append("")
    lines.append("=" * 40)
    lines.append("📈 週次サマリー")
    lines.append("=" * 40)

    # --- 週間パフォーマンスランキング ---
    lines.append("")
    lines.append("■ 週間パフォーマンス")
    lines.append("-" * 30)

    # 個別銘柄のみ（インデックス・ETFは別セクション）
    stock_tickers = [t for t in weekly if not t.startswith("^") and t != "SOXX"]
    index_tickers = [t for t in weekly if t.startswith("^") or t == "SOXX"]

    # 個別銘柄を週間変化率でソート
    sorted_stocks = sorted(
        stock_tickers, key=lambda t: weekly[t]["change"], reverse=True
    )

    lines.append("  [個別銘柄]")
    for rank, ticker in enumerate(sorted_stocks, 1):
        w = weekly[ticker]
        name = tickers.get(ticker, ticker)
        sign = "+" if w["change"] >= 0 else ""
        # 棒グラフ（正負対応）
        bar_len = min(abs(int(w["change"])), 20)
        if w["change"] >= 0:
            bar = "🟩" * bar_len
        else:
            bar = "🟥" * bar_len
        lines.append(
            f"  {rank}. {name:10s} {sign}{w['change']:6.2f}%  {bar}"
        )
        lines.append(
            f"     始値: {w['open']}  終値: {w['close']}"
            f"  高値: {w['high']}  安値: {w['low']}"
        )

    lines.append("")
    lines.append("  [指数・ETF]")
    for ticker in index_tickers:
        if ticker not in weekly:
            continue
        w = weekly[ticker]
        name = tickers.get(ticker, "VIX" if ticker == VIX_TICKER else ticker)
        sign = "+" if w["change"] >= 0 else ""
        lines.append(f"  {name:10s} {sign}{w['change']:6.2f}%  ({w['open']} → {w['close']})")

    # --- 週間ベスト/ワースト ---
    if sorted_stocks:
        best = sorted_stocks[0]
        worst = sorted_stocks[-1]
        lines.append("")
        lines.append("■ 週間ハイライト")
        lines.append("-" * 30)
        lines.append(
            f"  🏆 ベスト:  {tickers.get(best, best)} ({best})"
            f"  {weekly[best]['change']:+.2f}%"
        )
        lines.append(
            f"  📉 ワースト: {tickers.get(worst, worst)} ({worst})"
            f"  {weekly[worst]['change']:+.2f}%"
        )

    # --- 来週の注目イベント ---
    lines.append("")
    lines.append("■ 来週の注目イベント")
    lines.append("-" * 30)

    today = datetime.date.today()
    next_week_events = []

    # 決算（7〜14日後）
    for e in earnings_calendar:
        if 1 <= e["days_until"] <= 14:
            next_week_events.append(
                f"  📅 {e['name']} ({e['ticker']}) 決算"
                f"  {e['date'].strftime('%m/%d (%a)')}"
                f"  ({e['days_until']}日後)"
            )

    # FOMC（7〜14日後）
    for d in FOMC_DATES:
        days_until = (d - today).days
        if 1 <= days_until <= 14:
            next_week_events.append(
                f"  🏛 FOMC  {d.strftime('%m/%d (%a)')}  ({days_until}日後)"
            )

    if next_week_events:
        for ev in next_week_events:
            lines.append(ev)
    else:
        lines.append("  特になし")

    # --- 週間総評 ---
    lines.append("")
    lines.append("■ 週間総評")
    lines.append("-" * 30)

    # 全銘柄の平均週間変化
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


def calculate_score(
    report_data: dict, vix_data: dict | None, macro_data: dict
) -> tuple[int, bool]:
    score = 50
    risk_flag = False

    for ticker, d in report_data.items():
        change = d["change"]

        if ticker in ("NVDA", "AMD"):
            if change > 2:
                score += 5
            if d["current"] > d["ma50"]:
                score += 5
            if d["rsi"] < 30:
                score += 3
            if d["rsi"] > 70:
                score -= 3

        if ticker == "^GSPC":
            if change > 1:
                score += 5
            if change < -1:
                score -= 5

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
            score -= 15
            risk_flag = True
        elif vix_change > 5:
            score -= 10
            risk_flag = True

    tnx = macro_data.get("^TNX")
    if tnx:
        if tnx["change"] > 3:
            score -= 5
        elif tnx["change"] < -3:
            score += 3

    usdjpy = macro_data.get("JPY=X")
    if usdjpy:
        if usdjpy["change"] < -1.5:
            score -= 3

    gold = macro_data.get("GC=F")
    if gold:
        if gold["change"] > 2:
            score -= 3

    return int(clamp(score)), risk_flag


# ===== 温度ラベル =====


def get_temperature_label(score: int) -> str:
    if score >= 80:
        return "🔥 加速局面"
    elif score >= 65:
        return "🟢 強気"
    elif score >= 45:
        return "⚖ 中立"
    elif score >= 30:
        return "🟡 減速"
    else:
        return "❄ 崩れ"


# ===== ポジション配分 =====


def get_allocation(score: int, risk_flag: bool) -> dict:
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


# ===== 銘柄別配分 =====


def distribute(group: list[str], total_weight: float, report_data: dict) -> dict:
    strong, normal, reduced = [], [], []

    for ticker in group:
        d = report_data.get(ticker)
        if d is None:
            continue
        trend_ok = d["ma50"] > d["ma200"]
        rsi = d["rsi"]
        if d["change"] > 2 and trend_ok and 40 <= rsi <= 65:
            strong.append(ticker)
        elif rsi > 70:
            reduced.append(ticker)
        else:
            normal.append(ticker)

    active = [t for t in group if t in report_data]
    if not active:
        return {}
    if not strong and not normal and not reduced:
        per = round(total_weight / len(active), 1)
        return {t: per for t in active}

    if strong:
        weights = {"strong": 0.6, "normal": 0.3, "reduced": 0.1}
    else:
        weights = {"strong": 0.0, "normal": 0.8, "reduced": 0.2}

    result = {}
    for bucket, tickers_list in [
        ("strong", strong), ("normal", normal), ("reduced", reduced),
    ]:
        if tickers_list:
            per = round(total_weight * weights[bucket] / len(tickers_list), 1)
            for t in tickers_list:
                result[t] = per
    return result


def build_detailed_allocation(allocation: dict, report_data: dict) -> dict:
    detailed = {}
    detailed.update(
        distribute(semiconductor_stocks, allocation["semiconductor"], report_data)
    )
    detailed.update(
        distribute(ai_large_stocks, allocation["ai_large"], report_data)
    )
    return detailed


# ===== NVDAブースト =====


def apply_nvda_boost(
    detailed: dict, score: int, risk_flag: bool, report_data: dict
) -> dict:
    if "NVDA" not in detailed:
        return detailed
    boost = 0
    if score >= 65:
        boost += 5
    if risk_flag:
        boost -= 5
    if "NVDA" in report_data and report_data["NVDA"]["rsi"] < 35:
        boost += 3
    detailed["NVDA"] = max(0, detailed["NVDA"] + boost)
    return detailed


# ===== VIXボラティリティ調整 + 正規化 =====


def apply_vix_adjustment(
    detailed: dict, vix_data: dict | None, allocation: dict
) -> dict:
    vix_change = vix_data["change"] if vix_data else 0
    vol_factor = 1.0
    if vix_change > 5:
        vol_factor = 0.8
    elif vix_change < -3:
        vol_factor = 1.1

    for t in detailed:
        detailed[t] = round(detailed[t] * vol_factor, 1)

    total_weight = sum(detailed.values())
    target_total = allocation["semiconductor"] + allocation["ai_large"]
    if total_weight > 0:
        scale = target_total / total_weight
        for t in detailed:
            detailed[t] = round(detailed[t] * scale, 1)
    return detailed


# ===== ニュース取得（NewsAPI優先 → Google News RSSフォールバック） =====


def get_ai_news_from_newsapi() -> list[str]:
    """NewsAPI (everythingエンドポイント) からニュース取得"""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.info("NEWS_API_KEY 未設定、スキップ")
        return []

    # 過去3日分を取得（無料プランの制限内）
    from_date = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()

    url = (
        "https://newsapi.org/v2/everything?"
        "q=(AI OR semiconductor) AND (NVIDIA OR AMD OR Micron OR Broadcom)&"
        f"from={from_date}&"
        "language=en&sortBy=publishedAt&pageSize=5&"
        f"apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "ok":
            logger.warning(f"NewsAPI応答エラー: {data.get('message', 'unknown')}")
            return []

        titles = [
            a["title"] for a in data.get("articles", [])
            if a.get("title") and a["title"] != "[Removed]"
        ]
        return titles[:5]

    except Exception as e:
        logger.warning(f"NewsAPI取得失敗: {e}")
        return []


def get_ai_news_from_rss() -> list[str]:
    """Google News RSSからフォールバック取得"""
    queries = ["AI semiconductor", "NVIDIA AMD stock"]
    titles = []

    for query in queries:
        url = (
            "https://news.google.com/rss/search?"
            f"q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:3]:
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    titles.append(title_el.text)
        except Exception as e:
            logger.error(f"Google News RSS取得失敗 ({query}): {e}")

    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) >= 5:
            break
    return unique


def get_ai_news() -> list[str]:
    """NewsAPIを優先し、失敗時はGoogle News RSSにフォールバック"""
    news = get_ai_news_from_newsapi()
    if news:
        logger.info(f"NewsAPIから{len(news)}件取得")
        return news

    logger.info("NewsAPI失敗、Google News RSSにフォールバック")
    news = get_ai_news_from_rss()
    if news:
        logger.info(f"Google News RSSから{len(news)}件取得")
    else:
        logger.warning("ニュース取得: 全ソース失敗")
    return news


def analyze_news(news: list[str]) -> tuple[list[str], int]:
    translated = []
    negative_count = 0
    for title in news:
        lower = title.lower()
        for word in negative_keywords:
            if word in lower:
                negative_count += 1
        translated.append(translate_to_japanese(title))
    return translated, negative_count


# ===== リバランス判定 =====


def is_rebalance_day() -> bool:
    return datetime.date.today().weekday() == 4  # 金曜日


# ===== レポート生成 =====


def generate_report(
    score: int,
    temp: str,
    risk_flag: bool,
    report_data: dict,
    vix_data: dict | None,
    macro_data: dict,
    translated_news: list[str],
    negative_count: int,
    allocation: dict,
    detailed_allocation: dict,
    rebalance: bool,
    event_alerts: list[str],
    earnings_calendar: list[dict],
    fomc_upcoming: list[dict],
) -> str:
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

    # 決算カレンダー
    lines.append("")
    lines.append("■ 決算カレンダー（今後30日）")
    lines.append("-" * 30)
    upcoming_earnings = [e for e in earnings_calendar if 0 <= e["days_until"] <= 30]
    if upcoming_earnings:
        for e in upcoming_earnings:
            date_str = e["date"].strftime("%m/%d (%a)")
            days = e["days_until"]
            if days == 0:
                tag = "⚡本日"
            elif days <= 3:
                tag = f"🔴 {days}日後"
            elif days <= 7:
                tag = f"🟡 {days}日後"
            else:
                tag = f"   {days}日後"
            lines.append(f"  {tag}  {e['name']} ({e['ticker']})  {date_str}")
    else:
        lines.append("  今後30日以内の決算予定なし")

    # FOMC日程
    lines.append("")
    lines.append("■ FOMC日程")
    lines.append("-" * 30)
    if fomc_upcoming:
        for f in fomc_upcoming:
            date_str = f["date"].strftime("%m/%d (%a)")
            days = f["days_until"]
            if days == 0:
                tag = "⚡本日"
            elif days <= 3:
                tag = f"🔴 {days}日後"
            elif days <= 7:
                tag = f"🟡 {days}日後"
            else:
                tag = f"   {days}日後"
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
            if ticker == "^TNX":
                lines.append(
                    f"  {m['name']}: {m['current']}%"
                    f"  ({sign}{m['change']}%)"
                    f"  {diff_arrow(m['current'], m['prev'])}"
                )
            else:
                lines.append(
                    f"  {m['name']}: {m['current']}"
                    f"  ({sign}{m['change']}%)"
                    f"  {diff_arrow(m['current'], m['prev'])}"
                )

        warnings = []
        tnx = macro_data.get("^TNX")
        usdjpy = macro_data.get("JPY=X")
        gold = macro_data.get("GC=F")
        if tnx and tnx["change"] > 3:
            warnings.append("⚠ 金利急騰 → グロース株に逆風")
        if tnx and tnx["change"] < -3:
            warnings.append("✅ 金利低下 → グロース株に追い風")
        if usdjpy and usdjpy["change"] < -1.5:
            warnings.append("⚠ 急速な円高 → ドル建て資産目減り注意")
        if usdjpy and usdjpy["change"] > 1.5:
            warnings.append("✅ 円安進行 → ドル建て資産に追い風")
        if gold and gold["change"] > 2:
            warnings.append("⚠ 金価格急騰 → リスクオフの兆候")
        if warnings:
            lines.append("")
            for w in warnings:
                lines.append(f"  {w}")
    else:
        lines.append("  データ取得なし")

    # 銘柄テクニカル
    lines.append("")
    lines.append("■ 銘柄テクニカル")
    lines.append("-" * 30)

    all_tickers = {**tickers, VIX_TICKER: "VIX"}
    for ticker, name in all_tickers.items():
        d = report_data.get(ticker) if ticker != VIX_TICKER else vix_data
        if d is None:
            continue

        earnings_mark = ""
        for e in earnings_calendar:
            if e["ticker"] == ticker and 0 <= e["days_until"] <= EARNINGS_WARN_DAYS:
                earnings_mark = (
                    f" ⚡決算本日" if e["days_until"] == 0
                    else f" 📅決算{e['days_until']}日後"
                )
                break

        lines.append(f"  {name} ({ticker}){earnings_mark}")

        change_str = f"{d['change']:+.2f}%"
        if d.get("prev_change") is not None:
            momentum = d["change"] - d["prev_change"]
            if momentum > 0.5:
                momentum_icon = "📈 加速"
            elif momentum < -0.5:
                momentum_icon = "📉 減速"
            else:
                momentum_icon = "➡ 横ばい"
            lines.append(
                f"    前日比: {change_str}  (前日: {d['prev_change']:+.2f}%) {momentum_icon}"
            )
        else:
            lines.append(f"    前日比: {change_str}")

        lines.append(f"    MA50: {d['ma50']}  MA200: {d['ma200']}")

        rsi_str = f"{d['rsi']}"
        if d.get("rsi_prev") is not None:
            rsi_str += f"  ({diff_arrow(d['rsi'], d['rsi_prev'])})"
        lines.append(f"    RSI: {rsi_str}")

        vol_str = f"{d['volume_ratio']}倍" if d["volume_ratio"] is not None else "N/A"
        lines.append(f"    出来高倍率: {vol_str}")
        lines.append("")

    # ニュース
    lines.append("■ AI関連最新ニュース")
    lines.append("-" * 30)
    if translated_news:
        for n in translated_news:
            lines.append(f"  - {n}")
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
        bar_len = int(w / 2)
        bar = "█" * bar_len
        lines.append(f"  {t:6s}: {w:5.1f}%  {bar}")

    return "\n".join(lines)


# ===== メール送信 =====


def send_email(
    report: str,
    score: int,
    temp: str,
    risk_flag: bool,
    macro_data: dict,
    event_alerts: list[str],
    weekly_summary: str | None,
):
    gmail_user = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        logger.error("Gmail認証情報が設定されていません")
        return

    # 件名
    if event_alerts and any("決算まで" in a and "🔴" in a for a in event_alerts):
        subject = "📅⚠ 決算接近アラート + AI市場レポート"
    elif risk_flag:
        subject = "⚠ AI市場警戒アラート"
    elif weekly_summary:
        subject = "📈 週次サマリー + AI市場レポート"
    else:
        subject = "📊 Daily AI Market Report"

    # フルレポート（週次サマリーを結合）
    full_report = report
    if weekly_summary:
        full_report += "\n" + weekly_summary

    # マクロHTML
    macro_rows = ""
    for ticker, m in macro_data.items():
        sign = "+" if m["change"] >= 0 else ""
        color = "#e74c3c" if m["change"] < 0 else "#27ae60"
        val = f"{m['current']}%" if ticker == "^TNX" else f"{m['current']}"
        macro_rows += (
            f'<tr><td style="padding:6px;">{m["name"]}</td>'
            f'<td style="padding:6px;">{val}</td>'
            f'<td style="padding:6px;color:{color};">{sign}{m["change"]}%</td></tr>'
        )

    # イベントアラートHTML
    event_html = ""
    if event_alerts:
        alert_items = "".join(
            f'<li style="margin:4px 0;">{a}</li>' for a in event_alerts
        )
        event_html = f"""
        <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:12px;margin:15px 0;">
          <h3 style="margin:0 0 8px 0;">📅 イベントアラート</h3>
          <ul style="margin:0;padding-left:20px;">{alert_items}</ul>
        </div>
        """

    # 週次サマリーHTML
    weekly_html = ""
    if weekly_summary:
        weekly_html = f"""
        <div style="background:#e8f5e9;border:1px solid #4caf50;border-radius:8px;padding:12px;margin:15px 0;">
          <h3 style="margin:0 0 8px 0;">📈 週次サマリー</h3>
          <pre style="font-size:12px;line-height:1.5;margin:0;white-space:pre-wrap;">{weekly_summary}</pre>
        </div>
        """

    # スコアバーの色
    if score >= 65:
        score_color = "#27ae60"
    elif score >= 45:
        score_color = "#f39c12"
    else:
        score_color = "#e74c3c"

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

      {event_html}

      <h3>🌍 マクロ環境</h3>
      <table style="border-collapse:collapse;width:100%;">
        <tr style="background:#f5f5f5;">
          <th style="padding:8px;text-align:left;">指標</th>
          <th style="padding:8px;text-align:left;">現在値</th>
          <th style="padding:8px;text-align:left;">前日比</th>
        </tr>
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
    logger.info("===== AI市場監視Bot 起動 =====")

    # 1. テクニカルデータ取得
    report_data = {}
    for ticker in tickers:
        data = fetch_technical_data(ticker)
        if data:
            report_data[ticker] = data

    vix_data = fetch_technical_data(VIX_TICKER)

    # 2. マクロ指標取得
    macro_data = fetch_macro_data()

    # 3. 決算・イベントカレンダー取得
    earnings_calendar = fetch_earnings_calendar()
    fomc_upcoming = get_upcoming_fomc()
    event_alerts, event_score_adj = build_event_alerts(
        earnings_calendar, fomc_upcoming
    )

    # 4. スコアリング
    score, risk_flag = calculate_score(report_data, vix_data, macro_data)

    # 5. イベントによるスコア調整
    score = int(clamp(score + event_score_adj))

    # 6. ニュース分析
    news = get_ai_news()
    translated_news, negative_count = analyze_news(news)

    if negative_count >= 2:
        score = int(clamp(score - 10))
        risk_flag = True
    elif negative_count == 1:
        score = int(clamp(score - 5))

    # 7. 温度判定
    temp = get_temperature_label(score)

    # 8. ポジション配分
    allocation = get_allocation(score, risk_flag)

    # 9. リバランス判定
    rebalance = is_rebalance_day()

    # 10. 銘柄別詳細配分
    detailed_allocation = build_detailed_allocation(allocation, report_data)

    # 11. NVDAブースト
    detailed_allocation = apply_nvda_boost(
        detailed_allocation, score, risk_flag, report_data
    )

    # 12. VIX調整 + 正規化
    detailed_allocation = apply_vix_adjustment(
        detailed_allocation, vix_data, allocation
    )

    # 13. レポート生成
    report = generate_report(
        score, temp, risk_flag, report_data, vix_data, macro_data,
        translated_news, negative_count,
        allocation, detailed_allocation, rebalance,
        event_alerts, earnings_calendar, fomc_upcoming,
    )

    # 14. 週次サマリー（金曜のみ）
    weekly_summary = None
    if rebalance:
        logger.info("金曜日: 週次サマリー生成中...")
        weekly_data = fetch_weekly_performance()
        weekly_summary = generate_weekly_summary(
            weekly_data, score, temp, earnings_calendar
        )
        logger.info("週次サマリー生成完了")

    logger.info("\n" + report)
    if weekly_summary:
        logger.info("\n" + weekly_summary)

    # 15. メール送信
    send_email(
        report, score, temp, risk_flag, macro_data,
        event_alerts, weekly_summary,
    )

    logger.info("===== 処理完了 =====")


if __name__ == "__main__":
    main()
