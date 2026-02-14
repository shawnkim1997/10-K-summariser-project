"""
All-in-One Financial Analysis Dashboard — Hybrid Architecture
- Tab 1: 10-K & MD&A Insights (Item 7 + Item 1A → Gemini, qualitative only).
- Tab 2: 3-Scenario DCF Valuation (yfinance + sliders, no LLM).
- Tab 3: Industry Comps (yfinance multiples: Forward P/E, EV/EBITDA, P/B).
- Cost-effective: Gemini only for text; all numbers from yfinance.
"""

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Any


# Local prefs file for "Remember me" (API key & email). Path is in .gitignore.
_PREFS_PATH = Path(__file__).resolve().parent / ".app_prefs.json"
# Persistent 10-K cache: data/ticker_latest.json (Item 1A, 3, 7, 9A cleaned text).
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_prefs() -> dict:
    """Load saved API keys and email from local file. Keys: google_api_key, sec_email."""
    try:
        if _PREFS_PATH.exists():
            with open(_PREFS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_prefs(google_api_key: str, sec_email: str) -> None:
    """Save API keys and email to local file (only if user opted in)."""
    try:
        data = {
            "google_api_key": (google_api_key or "").strip(),
            "sec_email": (sec_email or "").strip(),
        }
        with open(_PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from yahooquery import Ticker as YQTicker
    from yahooquery import search as yq_search
except ImportError:
    YQTicker = None
    yq_search = None

# Company name → ticker for search/autocomplete (expand as needed)
COMPANY_LIST = [
    ("NVIDIA Corporation", "NVDA"), ("Apple Inc.", "AAPL"), ("Microsoft Corporation", "MSFT"),
    ("Amazon.com Inc.", "AMZN"), ("Alphabet Inc.", "GOOGL"), ("Meta Platforms Inc.", "META"),
    ("AMD", "AMD"), ("Intel Corporation", "INTC"), ("Qualcomm Inc.", "QCOM"), ("Tesla Inc.", "TSLA"),
    ("Berkshire Hathaway", "BRK.B"), ("JPMorgan Chase", "JPM"), ("Visa Inc.", "V"), ("UnitedHealth", "UNH"),
    ("Procter & Gamble", "PG"), ("Exxon Mobil", "XOM"), ("Johnson & Johnson", "JNJ"), ("Mastercard", "MA"),
    ("Chevron", "CVX"), ("Home Depot", "HD"), ("Merck", "MRK"), ("AbbVie", "ABBV"), ("Costco", "COST"),
    ("PepsiCo", "PEP"), ("Coca-Cola", "KO"), ("Pfizer", "PFE"), ("Walmart", "WMT"), ("Netflix", "NFLX"),
    ("Adobe", "ADBE"), ("Salesforce", "CRM"), ("Comcast", "CMCSA"), ("Cisco", "CSCO"), ("Oracle", "ORCL"),
    ("American Express", "AXP"), ("Bank of America", "BAC"), ("Wells Fargo", "WFC"), ("Verizon", "VZ"),
    ("AT&T", "T"), ("Walt Disney", "DIS"), ("Nike", "NKE"), ("McDonald's", "MCD"), ("Starbucks", "SBUX"),
    ("Goldman Sachs", "GS"), ("Morgan Stanley", "MS"), ("Target", "TGT"), ("Boeing", "BA"), ("IBM", "IBM"),
]
COMPANY_OPTIONS = [f"{t} - {n}" for n, t in COMPANY_LIST]
COMPANY_TICKER_MAP = {t: n for n, t in COMPANY_LIST}

MARKET_OPTIONS = [
    "US (S&P/Dow/Nasdaq)",
    "South Korea (KOSPI/KOSDAQ)",
    "Japan (Nikkei)",
    "UK (LSE)",
]


def get_global_ticker(ticker: str, market: str) -> str:
    """Format ticker for Yahoo Finance by market. US: as-is. South Korea: .KS or .KQ. Japan: .T. UK: .L. If ticker already has suffix, return as-is."""
    if not (ticker or "").strip():
        return (ticker or "").strip()
    t = (ticker or "").strip()
    if t.upper().endswith((".KS", ".KQ", ".T", ".L")):
        return t
    m = (market or "").strip()
    if "US" in m or not m:
        return t
    if "Korea" in m or "KOSPI" in m or "KOSDAQ" in m:
        return t + ".KS"
    if "Japan" in m or "Nikkei" in m:
        return t + ".T"
    if "UK" in m or "LSE" in m:
        return t + ".L"
    return t


def infer_market_from_ticker(ticker: str) -> str:
    """Infer market label from ticker suffix (for Deep-Dive routing when no Market selector)."""
    if not (ticker or "").strip():
        return MARKET_OPTIONS[0]
    t = (ticker or "").strip().upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "South Korea (KOSPI/KOSDAQ)"
    if t.endswith(".T"):
        return "Japan (Nikkei)"
    if t.endswith(".L"):
        return "UK (LSE)"
    return "US (S&P/Dow/Nasdaq)"


# Top-down sector analysis: industry → top 5 S&P 500 / NASDAQ tickers
SECTORS = {
    "Semiconductors & Hardware": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
    "Software & Cloud": ["MSFT", "ADBE", "CRM", "PANW", "CRWD"],
    "Consumer Retail": ["AMZN", "SBUX", "MCD", "WMT", "HD"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "V"],
    "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK"],
}


def get_edgar_downloader():
    from sec_edgar_downloader import Downloader
    return Downloader


def _slice_html_items_1a_to_9a(raw_html: str) -> str:
    """Fast string slice: keep only Item 1A through end of Item 9A to avoid parsing 50MB+ full file. Uses .find()/regex on raw string only."""
    if not raw_html or len(raw_html) < 5000:
        return raw_html
    start = -1
    for needle in ("Item 1A", "ITEM 1A", "Item 1a"):
        i = raw_html.find(needle)
        if i != -1 and (start == -1 or i < start):
            start = i
    if start == -1:
        m = re.search(r"Item\s+1A\s", raw_html, re.IGNORECASE)
        start = m.start() if m else 0
    else:
        start = max(0, start - 200)
    search_region = raw_html[start:]
    end_match = re.search(r"Item\s+10\s|Item\s+12\s|Part\s+III\b|PART\s+III\b", search_region, re.IGNORECASE)
    end = start + end_match.start() if end_match else len(raw_html)
    end = min(end, start + 8_000_000)
    return raw_html[start:end]


def _extract_text_from_html_string(html_str: str) -> str:
    """Parse HTML string with lxml; drop table/img/svg/style/script immediately to reduce memory and speed."""
    if not html_str or not html_str.strip():
        return ""
    try:
        soup = BeautifulSoup(html_str, "lxml")
    except Exception:
        soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup.find_all(["table", "img", "svg", "style", "script"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_html(html_path: Path) -> str:
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        with open(html_path, "r", encoding="latin-1", errors="replace") as f:
            raw = f.read()
    chunk = _slice_html_items_1a_to_9a(raw)
    return _extract_text_from_html_string(chunk)


def extract_text_from_file(file_path: Path) -> str:
    suf = file_path.suffix.lower()
    if suf in (".htm", ".html"):
        return extract_text_from_html(file_path)
    if suf == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text
    return ""


# Section patterns for 10-K items
ITEM1A_PATTERNS = [
    r"Item\s+1A\s*[.:]\s*Risk\s+Factors",
    r"ITEM\s+1A\s*[.:]\s*Risk\s+Factors",
]
ITEM7_PATTERNS = [
    r"Item\s+7\s*[.:]\s*Management['\u2019]s\s+Discussion\s+and\s+Analysis",
    r"ITEM\s+7\s*[.:]\s*Management['\u2019]s\s+Discussion",
    r"Item\s+7\s*[.:]\s*[\w\s]+MD&A",
]
ITEM8_PATTERNS = [
    r"Item\s+8\s*[.:]\s*Financial\s+Statements",
    r"ITEM\s+8\s*[.:]\s*Financial\s+Statements",
]
ITEM3_PATTERNS = [
    r"Item\s+3\s*[.:]\s*Legal\s+Proceedings",
    r"ITEM\s+3\s*[.:]\s*Legal\s+Proceedings",
]
ITEM9A_PATTERNS = [
    r"Item\s+9A\s*[.:]\s*Controls\s+and\s+Procedures",
    r"Item\s+9A\s*[.:]\s*Internal\s+Control",
    r"ITEM\s+9A\s*[.:]\s*Controls",
]


def _find_section_start(text: str, patterns: list, item_num: int) -> int:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.start()
    m = re.search(r"\bItem\s+" + str(item_num) + r"\b", text, re.IGNORECASE)
    return m.start() if m else -1


def find_item_section_generic(text: str, patterns: list, item_num: int, title_keywords: list, max_chars: int = 120000) -> str:
    start = _find_section_start(text, patterns, item_num)
    if start == -1:
        pattern = re.compile(
            r"\bItem\s+" + str(item_num) + r"\b[.\s]*[^\n]*(" + "|".join(re.escape(k) for k in title_keywords) + r")?",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return ""
        start = match.start()
    next_item = re.search(r"\n\s*Item\s+\d+[A-Z]?\s+", text[start + 100:], re.IGNORECASE)
    if next_item:
        end = start + 100 + next_item.start()
    else:
        end = min(start + max_chars, len(text))
    return text[start:end].strip()


def clean_text_for_llm(html_content: str) -> str:
    """Aggressive cleaning for LLM: strip tables/code, collapse whitespace, drop non-ASCII. Uses lxml for speed; drops table/img/svg/style/script."""
    if not html_content or not html_content.strip():
        return ""
    try:
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup.find_all(["table", "img", "style", "script", "svg", "math"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_content)
    text = re.sub(r"\s+", " ", text)
    text = " ".join(text.split())
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line) or re.fullmatch(r"[\.\-\s\-]+", line):
            continue
        if re.match(r"^(page\s+\d+|\d+)\s*$", line, re.IGNORECASE) and len(line) < 20:
            continue
        lines.append(line)
    result = " ".join(lines)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def smart_chunk(section: str, max_chars: int = 10000, head_ratio: float = 0.5) -> str:
    """Limit payload for Gemini; 10k chars ≈ 2.5k tokens for fast response."""
    if not section or len(section) <= max_chars:
        return section
    head_size = int(max_chars * head_ratio)
    tail_size = max_chars - head_size - 100
    return section[:head_size] + " [ ... middle omitted ... ] " + section[-tail_size:]


def find_downloaded_10k_path(download_root: Path, ticker: str) -> Optional[Path]:
    ticker_upper = ticker.upper()
    for base in (download_root / "sec-edgar-filings", download_root):
        path_10k = base / ticker_upper / "10-K"
        if path_10k.exists():
            subdirs = sorted([d for d in path_10k.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if subdirs:
                return subdirs[0]
    for base in (download_root / "sec-edgar-filings", download_root):
        if not base.exists():
            continue
        for company_dir in base.iterdir():
            if not company_dir.is_dir():
                continue
            path_10k = company_dir / "10-K"
            if path_10k.exists():
                subdirs = sorted([d for d in path_10k.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if subdirs:
                    return subdirs[0]
    return None


def find_all_10k_filing_dirs(download_root: Path, ticker: str) -> list:
    """Return list of 10-K filing dirs sorted newest first (for multi-year comparison)."""
    ticker_upper = ticker.upper()
    for base in (download_root / "sec-edgar-filings", download_root):
        path_10k = base / ticker_upper / "10-K"
        if path_10k.exists():
            subdirs = sorted([d for d in path_10k.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            return subdirs
    return []


def get_main_10k_text(filing_dir: Path) -> str:
    all_text = []
    for ext in ("*.htm", "*.html", "*.txt"):
        for path in filing_dir.rglob(ext):
            try:
                t = extract_text_from_file(path)
                if len(t) > 1000:
                    all_text.append((path, t))
            except Exception:
                continue
    if not all_text:
        return ""
    _, main_text = max(all_text, key=lambda x: len(x[1]))
    return main_text


def _get_10k_cache_path(ticker: str) -> Path:
    """Path for cached 10-K sections: data/TICKER_latest.json."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR / f"{ticker.upper()}_latest.json"


def _load_10k_from_cache(ticker: str) -> Optional[dict]:
    """Load Item 1A, 3, 7, 8, 9A (plain text) from data/ticker_latest.json. Returns None if missing. When present, no re-download or re-parse."""
    path = _get_10k_cache_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_10k_to_cache(ticker: str, data: dict) -> None:
    """Save cleaned 10-K sections to data/ticker_latest.json."""
    path = _get_10k_cache_path(ticker)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)


def _extract_item_from_full(text: str, patterns: list, item_num: int, keywords: list, max_chars: int = 60000) -> str:
    """Extract one item section from full 10-K text."""
    start = _find_section_start(text, patterns, item_num)
    if start < 0:
        pattern = re.compile(r"\bItem\s+" + str(item_num) + r"[A-Z]?\b[.\s]*[^\n]*", re.IGNORECASE)
        match = pattern.search(text)
        start = match.start() if match else -1
    if start < 0:
        return ""
    next_item = re.search(r"\n\s*Item\s+\d+[A-Z]?\s+", text[start + 100:], re.IGNORECASE)
    end = start + 100 + next_item.start() if next_item else min(start + max_chars, len(text))
    return text[start:end].strip()


def download_and_extract_all_items(ticker: str, email: str) -> dict:
    """Download latest 10-K, extract Item 1A, 3, 7, 9A; clean and return (and optionally cache)."""
    Downloader = get_edgar_downloader()
    with tempfile.TemporaryDirectory() as tmpdir:
        download_root = Path(tmpdir)
        dl = Downloader("FQDC-10K-Analyzer", email, str(download_root))
        dl.get("10-K", ticker.upper(), limit=1, download_details=True)
        filing_dir = find_downloaded_10k_path(download_root, ticker)
        if not filing_dir:
            raise FileNotFoundError(f"Could not find 10-K for ticker '{ticker}'.")
        full_text = get_main_10k_text(filing_dir)
        if not full_text:
            raise ValueError("Could not extract text from the 10-K.")
    item1a = find_item_section_generic(full_text, ITEM1A_PATTERNS, 1, ["Risk", "Factors"], max_chars=80000)
    item3 = _extract_item_from_full(full_text, ITEM3_PATTERNS, 3, ["Legal", "Proceedings"], max_chars=40000)
    item9a = _extract_item_from_full(full_text, ITEM9A_PATTERNS, 9, ["Controls", "Procedures", "Internal"], max_chars=40000)
    start7 = _find_section_start(full_text, ITEM7_PATTERNS, 7)
    text_after_7 = full_text[start7:] if start7 >= 0 else full_text
    item7 = find_item_section_generic(text_after_7, ITEM7_PATTERNS, 7, ["Management's Discussion", "MD&A", "Analysis"], max_chars=100000)
    if not item7 and text_after_7:
        item7 = text_after_7[:120000]
    item8 = _extract_item_from_full(full_text, ITEM8_PATTERNS, 8, ["Financial Statements", "Supplementary Data"], max_chars=200000)
    data = {
        "item1a": clean_text_for_llm(item1a or ""),
        "item3": clean_text_for_llm(item3 or ""),
        "item9a": clean_text_for_llm(item9a or ""),
        "item7": clean_text_for_llm(item7 or ""),
        "item8": clean_text_for_llm(item8 or ""),
    }
    _save_10k_to_cache(ticker, data)
    return data


def get_10k_sections(ticker: str, email: str) -> tuple[dict, str]:
    """Return (sections dict, status). status = 'cache' if loaded from file else 'downloaded'. Cached ticker skips download and parsing entirely (item1a, item3, item7, item8, item9a)."""
    cached = _load_10k_from_cache(ticker)
    if cached is not None:
        return cached, "cache"
    return download_and_extract_all_items(ticker, email), "downloaded"


def download_and_extract_item7_and_1a(ticker: str, email: str) -> tuple[str, str, str]:
    """Fetch 10-K from SEC EDGAR and return full_text, Item 1A (Risk Factors), Item 7 (MD&A). Uses cache when available."""
    sections, _ = get_10k_sections(ticker, email)
    return "", sections.get("item1a", "") or "", sections.get("item7", "") or ""


def download_item7_latest_and_3y_ago(ticker: str, email: str) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Download up to 5 10-Ks; extract Item 1A (latest only) and Item 7 from latest and from 3 years ago.
    Returns (item1a_latest, item7_latest, item7_3y_ago, has_comparison). If < 4 filings, item7_3y_ago is None."""
    Downloader = get_edgar_downloader()
    with tempfile.TemporaryDirectory() as tmpdir:
        download_root = Path(tmpdir)
        dl = Downloader("FQDC-10K-Analyzer", email, str(download_root))
        dl.get("10-K", ticker.upper(), limit=5, download_details=True)
        filing_dirs = find_all_10k_filing_dirs(download_root, ticker)
        if not filing_dirs:
            raise FileNotFoundError(f"Could not find 10-K for ticker '{ticker}'.")
        full_latest = get_main_10k_text(filing_dirs[0])
        if not full_latest:
            raise ValueError("Could not extract text from the latest 10-K.")
        item1a = find_item_section_generic(
            full_latest, ITEM1A_PATTERNS, 1, ["Risk", "Factors"], max_chars=80000
        )
        text_after_7 = full_latest[_find_section_start(full_latest, ITEM7_PATTERNS, 7):] if _find_section_start(full_latest, ITEM7_PATTERNS, 7) >= 0 else full_latest
        item7_latest = find_item_section_generic(
            text_after_7, ITEM7_PATTERNS, 7, ["Management's Discussion", "MD&A", "Analysis"], max_chars=100000
        )
        if not item7_latest and text_after_7:
            item7_latest = smart_chunk(text_after_7[:120000], max_chars=20000)
        item7_3y_ago = None
        has_comparison = False
        if len(filing_dirs) >= 4:
            full_3y = get_main_10k_text(filing_dirs[3])
            if full_3y:
                text_3y = full_3y[_find_section_start(full_3y, ITEM7_PATTERNS, 7):] if _find_section_start(full_3y, ITEM7_PATTERNS, 7) >= 0 else full_3y
                item7_3y_ago = find_item_section_generic(
                    text_3y, ITEM7_PATTERNS, 7, ["Management's Discussion", "MD&A", "Analysis"], max_chars=100000
                )
                if not item7_3y_ago and text_3y:
                    item7_3y_ago = smart_chunk(text_3y[:120000], max_chars=20000)
                has_comparison = bool(item7_3y_ago)
    return item1a or "", item7_latest or "", item7_3y_ago, has_comparison


# ---------- Gemini (qualitative only) ----------
GEMINI_MODEL = "gemini-2.0-flash"
RATE_LIMIT_WAIT_SEC = 60


def get_gemini_model(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def _is_rate_limit_error(e: Exception) -> bool:
    err_msg = str(e).lower()
    return "429" in err_msg or "resourcelimited" in err_msg or "resource exhausted" in err_msg or getattr(e, "code", None) == 429


def _generate_with_retry(model, content, config, max_retries: int = 3):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return model.generate_content(content, generation_config=config)
        except Exception as e:
            last_err = e
            if attempt < max_retries and _is_rate_limit_error(e):
                time.sleep(RATE_LIMIT_WAIT_SEC)
                continue
            raise
    raise last_err


def _generate_stream(model, content, config):
    """Yield text chunks from Gemini with stream=True. For use with st.write_stream()."""
    try:
        response = model.generate_content(content, generation_config=config, stream=True)
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception:
        raise


def _split_into_chunks(text: str, max_chars: int = 22000, min_chunk: int = 5000) -> list:
    """Split text into sequential chunks without cutting mid-sentence when possible."""
    if not text or len(text) <= max_chars:
        return [text] if text and text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            break_at = text.rfind("\n\n", start, end + 1)
            if break_at > start + min_chunk:
                end = break_at + 2
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _gemini_summarize_segment(api_key: str, segment_text: str, ticker: str, segment_label: str) -> str:
    """Extract strategic shifts and hidden risks from one segment. No trimming."""
    model = get_gemini_model(api_key)
    prompt = f"""You are a senior equity analyst. The following is one segment of the 10-K for {ticker} (Item 1A Risk Factors and/or Item 7 MD&A).
Extract and list all significant: (1) strategic shifts or priorities, (2) hidden or material risks, (3) management tone cues. Use concise bullet points. Do not omit important details. Segment: {segment_label}."""
    full = f"""--- 10-K Segment ---\n\n{segment_text[:50000]}\n\n---\n\n{prompt}"""
    try:
        r = _generate_with_retry(model, full, {"temperature": 0.2, "max_output_tokens": 2048})
        return (r.text or "").strip()
    except Exception:
        return ""


def _gemini_synthesize_report(api_key: str, segment_summaries: list, ticker: str, sector: str, industry: str) -> str:
    """Synthesis call: turn segment summaries into Executive Insight Report."""
    model = get_gemini_model(api_key)
    combined = "\n\n---\n\n".join(segment_summaries)
    kpi_note = f" Sector: {sector}; Industry: {industry}. Include industry-specific KPIs if mentioned." if sector and sector != "N/A" else ""
    prompt = f"""You are a senior equity analyst. Use British English. Below are summarized insights from the full 10-K for {ticker} (Item 1A and Item 7). Create the final **Executive Insight Report** with these sections:

1. **Management's Tone (Sentiment)**: Overall tone and supporting evidence.
2. **Current Strategy & Priorities**: Key strategic focus, capital allocation, growth drivers.
3. **Major Hidden Risks**: The 3–4 most material risks investors might overlook.
4. **Forensic / Quality of Earnings**: Accounting caveats, one-offs, cash flow vs earnings. If none material, say so briefly.{kpi_note}

Use clear headings. Do not invent figures. Keep under 900 words."""
    full = f"""--- Segment Summaries ---\n\n{combined}\n\n---\n\n{prompt}"""
    try:
        r = _generate_with_retry(model, full, {"temperature": 0.3, "max_output_tokens": 4096})
        return (r.text or "").strip()
    except Exception:
        return ""


def _gemini_forensic_audit(api_key: str, item3: str, item9a: str, ticker: str) -> str:
    """Dedicated high-priority check: Material Weaknesses, lawsuits, off-balance-sheet from Item 3 and 9A."""
    model = get_gemini_model(api_key)
    combined = (item3 or "") + "\n\n---\n\n" + (item9a or "")
    if not combined.strip():
        return "✅ No Item 3 / 9A text provided; skip forensic."
    prompt = f"""From the following 10-K excerpts for {ticker} (Item 3 Legal Proceedings and Item 9A Controls/Internal Control), list any:
- Material weaknesses in internal control
- Significant legal proceedings or litigation
- Off-balance-sheet or governance red flags
If none of the above, output exactly: "✅ No material red flags or special issues detected in Item 3 and 9A."
Be concise (under 150 words)."""
    full = f"""--- Item 3 & 9A ---\n\n{combined[:30000]}\n\n---\n\n{prompt}"""
    try:
        r = _generate_with_retry(model, full, {"temperature": 0.1, "max_output_tokens": 512})
        return (r.text or "").strip()
    except Exception:
        return ""


_REQUIRED_FINANCIAL_KEYS = [
    "Revenue", "CostOfRevenue", "OperatingExpenses", "NetIncome",
    "TotalAssets", "CurrentAssets", "CurrentLiabilities", "LongTermDebt",
    "OperatingCashFlow", "SharesOutstanding",
]


@st.cache_data(ttl=3600)
def get_sec_financials_llm(api_key: str, item8_text: str, ticker: str) -> dict:
    """Extract Current Year and Previous Year financial figures from 10-K Item 8 via Gemini. Returns dict with current_yr and previous_yr (each with 10 numeric fields). Cached by (api_key, item8_text, ticker)."""
    if not (api_key or "").strip() or not (item8_text or "").strip():
        return {}
    payload = smart_chunk((item8_text or "").strip(), max_chars=35000)
    model = get_gemini_model(api_key)
    prompt = f"""You are a financial analyst. Below is Item 8 (Financial Statements and Supplementary Data) from the latest 10-K for {ticker}.

Extract the following figures for the **Current Year** (most recent fiscal year) and **Previous Year** (prior fiscal year). Use the exact numbers from the financial statements. All monetary values in millions (e.g. 50000 for $50 billion). Shares in millions.

Return ONLY a valid JSON object, no other text. Use this exact structure:
{{
  "current_yr": {{
    "Revenue": <number>,
    "CostOfRevenue": <number>,
    "OperatingExpenses": <number>,
    "NetIncome": <number>,
    "TotalAssets": <number>,
    "CurrentAssets": <number>,
    "CurrentLiabilities": <number>,
    "LongTermDebt": <number>,
    "OperatingCashFlow": <number>,
    "SharesOutstanding": <number>
  }},
  "previous_yr": {{
    "Revenue": <number>,
    "CostOfRevenue": <number>,
    "OperatingExpenses": <number>,
    "NetIncome": <number>,
    "TotalAssets": <number>,
    "CurrentAssets": <number>,
    "CurrentLiabilities": <number>,
    "LongTermDebt": <number>,
    "OperatingCashFlow": <number>,
    "SharesOutstanding": <number>
  }}
}}

If a value is not found in the document, use 0 or a reasonable estimate and still include the key. Output nothing except this JSON."""

    full = f"""--- Item 8 (Financial Statements) ---\n\n{payload}\n\n---\n\n{prompt}"""
    try:
        r = _generate_with_retry(model, full, {"temperature": 0.0, "max_output_tokens": 2048})
        raw = (r.text or "").strip()
        if not raw:
            return {}
        raw = re.sub(r"^```\s*json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
        raw = raw.strip()
        out = json.loads(raw)
        cur = out.get("current_yr") or {}
        prev = out.get("previous_yr") or {}
        for key in _REQUIRED_FINANCIAL_KEYS:
            cur[key] = _safe_float(cur.get(key)) or 0
            prev[key] = _safe_float(prev.get(key)) or 0
        return {"current_yr": cur, "previous_yr": prev}
    except (json.JSONDecodeError, Exception):
        return {}


def get_gemini_item7_strategy(api_key: str, item7_text: str, ticker: str, sector: str, industry: str) -> str:
    """Item 7 only: business performance, strategic shifts, capital allocation."""
    if not (item7_text or "").strip():
        return "No Item 7 (MD&A) text available."
    model = get_gemini_model(api_key)
    text = smart_chunk(clean_text_for_llm(item7_text), max_chars=10000)
    sector_note = f" Sector: {sector}; Industry: {industry}." if sector and sector != "N/A" else ""
    prompt = f"""You are a senior equity analyst. Use British English. The text below is **Item 7 (Management's Discussion and Analysis)** from the latest 10-K for {ticker}.{sector_note}

Provide a concise **Management Strategy** report with these sections:

1. **Business performance**: Key revenue, margin, or segment highlights management emphasises.
2. **Strategic shifts**: Changes in priorities, growth drivers, or capital allocation (e.g. capex, M&A, buybacks).
3. **Capital allocation**: How management describes use of cash (dividends, debt paydown, R&D, acquisitions).

Use clear headings. Do not invent figures. Keep under 600 words. Focus only on narrative insights; ignore missing quantitative data.
Even if the source text is in another language (e.g. Korean or Japanese), analyse it and output your final report strictly in British English."""
    full = f"""--- Item 7 (MD&A) ---\n\n{text}\n\n---\n\n{prompt}"""
    try:
        r = _generate_with_retry(model, full, {"temperature": 0.3, "max_output_tokens": 2048})
        return (r.text or "").strip()
    except Exception:
        return ""


def get_gemini_item7_strategy_stream(api_key: str, item7_text: str, ticker: str, sector: str, industry: str):
    """Generator that yields MD&A strategy report chunks for real-time streaming (e.g. st.write_stream)."""
    if not (item7_text or "").strip():
        yield "No Item 7 (MD&A) text available."
        return
    model = get_gemini_model(api_key)
    text = smart_chunk(clean_text_for_llm(item7_text), max_chars=10000)
    sector_note = f" Sector: {sector}; Industry: {industry}." if sector and sector != "N/A" else ""
    prompt = f"""You are a senior equity analyst. Use British English. The text below is **Item 7 (Management's Discussion and Analysis)** from the latest 10-K for {ticker}.{sector_note}

Provide a concise **Management Strategy** report with these sections:

1. **Business performance**: Key revenue, margin, or segment highlights management emphasises.
2. **Strategic shifts**: Changes in priorities, growth drivers, or capital allocation (e.g. capex, M&A, buybacks).
3. **Capital allocation**: How management describes use of cash (dividends, debt paydown, R&D, acquisitions).

Use clear headings. Do not invent figures. Keep under 600 words. Focus only on narrative insights; ignore missing quantitative data.
Even if the source text is in another language (e.g. Korean or Japanese), analyse it and output your final report strictly in British English."""
    full = f"""--- Item 7 (MD&A) ---\n\n{text}\n\n---\n\n{prompt}"""
    config = {"temperature": 0.3, "max_output_tokens": 2048}
    yield from _generate_stream(model, full, config)


def get_gemini_item1a_risks(api_key: str, item1a_text: str, item3: str, item9a: str, ticker: str) -> str:
    """Item 1A only: legal, operational, market-related threats. Includes Forensic Audit (Item 3 & 9A) as safety check."""
    if not (item1a_text or "").strip():
        return "No Item 1A (Risk Factors) text available."
    model = get_gemini_model(api_key)
    text = smart_chunk(clean_text_for_llm(item1a_text), max_chars=10000)
    prompt = f"""You are a senior equity analyst. Use British English. The text below is **Item 1A (Risk Factors)** from the latest 10-K for {ticker}.

Provide a concise **Risk Factors** report with these sections:

1. **Legal & regulatory risks**: Litigation, regulatory changes, compliance.
2. **Operational risks**: Supply chain, key person, technology, execution.
3. **Market & competitive risks**: Demand, competition, macro, currency.

Use clear headings. Do not invent figures. Keep under 500 words. Focus only on narrative insights; ignore missing quantitative data.
Even if the source text is in another language (e.g. Korean or Japanese), analyse it and output your final report strictly in British English."""
    full = f"""--- Item 1A (Risk Factors) ---\n\n{text}\n\n---\n\n{prompt}"""
    try:
        report = _generate_with_retry(model, full, {"temperature": 0.3, "max_output_tokens": 2048})
        risks = (report.text or "").strip()
    except Exception:
        risks = ""
    forensic = _gemini_forensic_audit(api_key, item3 or "", item9a or "", ticker)
    return (risks or "") + "\n\n---\n\n**Forensic Audit (Item 3 & 9A)**\n\n" + (forensic or "")


def get_gemini_item1a_risks_stream(api_key: str, item1a_text: str, ticker: str):
    """Generator that yields Risk Factors report chunks for real-time streaming. Caller appends Forensic (Item 3 & 9A) after stream."""
    if not (item1a_text or "").strip():
        yield "No Item 1A (Risk Factors) text available."
        return
    model = get_gemini_model(api_key)
    text = smart_chunk(clean_text_for_llm(item1a_text), max_chars=10000)
    prompt = f"""You are a senior equity analyst. Use British English. The text below is **Item 1A (Risk Factors)** from the latest 10-K for {ticker}.

Provide a concise **Risk Factors** report with these sections:

1. **Legal & regulatory risks**: Litigation, regulatory changes, compliance.
2. **Operational risks**: Supply chain, key person, technology, execution.
3. **Market & competitive risks**: Demand, competition, macro, currency.

Use clear headings. Do not invent figures. Keep under 500 words. Focus only on narrative insights; ignore missing quantitative data.
Even if the source text is in another language (e.g. Korean or Japanese), analyse it and output your final report strictly in British English."""
    full = f"""--- Item 1A (Risk Factors) ---\n\n{text}\n\n---\n\n{prompt}"""
    config = {"temperature": 0.3, "max_output_tokens": 2048}
    yield from _generate_stream(model, full, config)


def get_mda_chunked_insights(
    api_key: str, sections: dict, ticker: str, sector: str, industry: str, progress_callback=None
) -> str:
    """Full-text analysis: chunk 1A+7, summarize each segment, synthesize report; then append forensic (Item 3, 9A). progress_callback(step: str) optional."""
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)
    combined = (sections.get("item1a") or "") + "\n\n---\n\n" + (sections.get("item7") or "")
    combined = combined.strip()
    if not combined:
        return "No 10-K text available to analyse."
    chunks = _split_into_chunks(combined, max_chars=22000)
    if not chunks:
        return "No content extracted."
    summaries = []
    n = len(chunks)
    for i, ch in enumerate(chunks):
        _progress(f"Analyzing Segment {i+1}/{n}...")
        summary = _gemini_summarize_segment(api_key, ch, ticker, f"Segment {i+1}/{n}")
        if summary:
            summaries.append(summary)
    if not summaries:
        return "Segment analysis produced no summaries."
    _progress("Synthesizing final report...")
    report = _gemini_synthesize_report(api_key, summaries, ticker, sector or "N/A", industry or "N/A")
    _progress("Running forensic audit (Item 3 & 9A)...")
    forensic = _gemini_forensic_audit(api_key, sections.get("item3") or "", sections.get("item9a") or "", ticker)
    return (report or "") + "\n\n---\n\n**Forensic (Item 3 & 9A)**\n\n" + (forensic or "")


def get_mda_insights(api_key: str, item1a_text: str, item7_text: str, ticker: str) -> str:
    """Send Item 1A + Item 7 to Gemini. Analyse: 1) Management's Tone (Sentiment), 2) Key Strategic Shifts, 3) Major Hidden Risks."""
    model = get_gemini_model(api_key)
    combined = []
    if item1a_text:
        combined.append(clean_text_for_llm(item1a_text))
    if item7_text:
        combined.append(clean_text_for_llm(item7_text))
    combined_text = "\n\n---\n\n".join(combined)
    combined_text = smart_chunk(combined_text, max_chars=22000)

    user_prompt = f"""You are a senior equity analyst. Use British English.

The text below is from the 10-K for {ticker}: **Item 1A (Risk Factors)** and **Item 7 (Management's Discussion and Analysis)**. HTML has been stripped; analyse only the substance.

Provide a concise report with three sections:

1. **Management's Tone (Sentiment)**: Is the overall tone positive, cautious, or negative? Quote 1–2 short phrases that support your view.

2. **Key Strategic Shifts**: What strategic priorities or shifts does management emphasise (e.g. capital allocation, growth drivers, new segments)? Be specific.

3. **Major Hidden Risks**: From both Risk Factors and MD&A, what are the 3–4 most material risks that an investor might overlook? Cite the document.

Use clear headings. Do not invent figures. Keep the response focused and under 800 words."""

    full_content = f"""--- 10-K Excerpt (Item 1A + Item 7) ---\n\n{combined_text}\n\n---\n\n{user_prompt}"""

    try:
        response = _generate_with_retry(
            model, full_content, {"temperature": 0.3, "max_output_tokens": 4096}
        )
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise
    if not response or not response.text:
        return "No analysis generated."
    return response.text.strip()


def get_mda_comparative_insights(
    api_key: str,
    item1a_text: str,
    item7_latest: str,
    item7_3y_ago: Optional[str],
    ticker: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
) -> str:
    """Comparative analysis: if item7_3y_ago provided, compare MD&As over 3 years; else single-year. Sector-aware: extract industry-specific Non-GAAP KPIs."""
    model = get_gemini_model(api_key)
    sector_label = (sector or "N/A").strip()
    industry_label = (industry or "N/A").strip()
    kpi_instruction = (
        f" Given that this company is in the **{sector_label}** sector"
        + (f" (industry: {industry_label})" if industry_label != "N/A" else "")
        + ", meticulously scan the MD&A to find and extract **industry-specific Non-GAAP KPIs** "
        "(e.g. Same-Store Sales Growth for Retail, ARR/NDR for Software, DAU/MAU for Tech). Present these hidden KPIs in a **clean markdown table** with columns such as KPI name, value, and period if stated."
    )
    if not item7_3y_ago or not item7_3y_ago.strip():
        combined = []
        if item1a_text:
            combined.append(clean_text_for_llm(item1a_text))
        if item7_latest:
            combined.append(clean_text_for_llm(item7_latest))
        combined_text = "\n\n---\n\n".join(combined)
        combined_text = smart_chunk(combined_text, max_chars=22000)
        user_prompt = f"""You are a senior equity analyst. Use British English.
The text below is from the **latest 10-K only** for {ticker}: **Item 1A (Risk Factors)** and **Item 7 (MD&A)**. Provide a focused deep-dive report:

1. **Management's Tone (Sentiment)**: Overall tone and 1–2 supporting phrases.
2. **Current Strategy & Priorities**: Key strategic focus, capital allocation, growth drivers from this filing only.
3. **Major Hidden Risks**: From Item 1A and MD&A, the 3–4 most material risks investors might overlook.
4. **Forensic / Quality of Earnings**: Any red flags in MD&A (accounting caveats, one-offs, cash flow vs earnings, segment disclosure). If none material, say so briefly.{kpi_instruction}
**Token-saving (Item 3 / 9A):** If no material weaknesses, major lawsuits, or off-balance-sheet red flags, output exactly: "✅ No material red flags or special issues detected in Item 3 and 9A."
Use clear headings. Under 800 words."""
        full_content = f"""--- 10-K Excerpt (Latest Year) ---\n\n{combined_text}\n\n---\n\n{user_prompt}"""
    else:
        latest_clean = smart_chunk(clean_text_for_llm(item7_latest), max_chars=12000)
        past_clean = smart_chunk(clean_text_for_llm(item7_3y_ago), max_chars=12000)
        user_prompt = f"""You are a senior equity analyst. Use British English.
Below are **Item 7 (Management's Discussion and Analysis)** from the 10-K for {ticker}: **LATEST YEAR** and **THREE YEARS AGO**. Perform a **Comparative Analysis**.

1. **Core strategy**: What has changed in the company's stated strategy, priorities, or capital allocation between then and now?
2. **Emerging risks**: What new risks appear in the latest MD&A that were absent or less prominent 3 years ago?
3. **Management's tone**: How has the overall tone (confidence, caution, optimism) shifted? Quote 1–2 phrases from each period if relevant.
4. **Industry-specific KPIs**:{kpi_instruction}
5. **Item 3 (Legal) & Item 9A (Internal Controls):** You must save output tokens. If there are no material weaknesses, no massive lawsuits, and no major off-balance sheet red flags, DO NOT generate a long explanation. Simply output exactly: "✅ No material red flags or special issues detected in Item 3 and 9A." and move on.

Use clear headings. Do not invent figures. Keep the response focused and under 900 words."""
        full_content = f"""--- MD&A LATEST YEAR ---\n\n{latest_clean}\n\n--- MD&A THREE YEARS AGO ---\n\n{past_clean}\n\n---\n\n{user_prompt}"""
    try:
        response = _generate_with_retry(
            model, full_content, {"temperature": 0.3, "max_output_tokens": 4096}
        )
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise
    if not response or not response.text:
        return "No analysis generated."
    return response.text.strip()


def _run_mda_analysis_background(ticker: str, api_key: str, sec_email: str) -> None:
    """Run download + Gemini in background (latest 10-K only for speed). Store result or error in st.session_state."""
    try:
        _, item1a, item7_latest = download_and_extract_item7_and_1a(ticker, sec_email)
        si = get_sector_industry(ticker)
        analysis = get_mda_comparative_insights(
            api_key, item1a or "", item7_latest or "", None, ticker,
            sector=si.get("sector"), industry=si.get("industry"),
        )
        st.session_state["mda_analysis_result"] = analysis
        st.session_state["mda_analysis_excerpt"] = ((item1a or "") + "\n\n---\n\n" + (item7_latest or ""))[:12000]
        st.session_state["mda_analysis_error"] = None
    except Exception as e:
        st.session_state["mda_analysis_error"] = str(e)
        st.session_state["mda_analysis_result"] = None
        st.session_state["mda_analysis_excerpt"] = None
    finally:
        st.session_state["mda_analysis_running"] = False
        st.session_state["mda_analysis_done"] = True
        st.session_state["mda_analysis_ticker"] = ticker


def get_industry_outlook(api_key: str, industry_name: str, tickers: list) -> str:
    """Gemini: Wall Street macro analyst-style Industry Outlook for the selected sector (12–18 months)."""
    model = get_gemini_model(api_key)
    ticker_list_str = ", ".join(str(t).upper() for t in tickers if t)
    user_prompt = f"""Act as an elite Wall Street macro analyst. Provide a concise **Industry Outlook** report for the **{industry_name}** sector, which includes leading companies like {ticker_list_str}.

Focus on:
1. **Macro trends** affecting this industry over the next 12–18 months.
2. **Major growth drivers** (e.g., AI, interest rates, consumer spending, regulation).
3. **Key headwinds or regulatory risks** that could impact valuations or growth.

Use clear headings. Be specific but concise. Keep the response under 600 words."""
    full_content = user_prompt
    try:
        response = _generate_with_retry(
            model, full_content, {"temperature": 0.4, "max_output_tokens": 2048}
        )
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise
    if not response or not response.text:
        return "No industry outlook generated."
    return response.text.strip()


# ---------- Financial data helpers (yahooquery primary, yfinance fallback) ----------
def _safe_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and (x != x or pd.isna(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------- yahooquery: map to our index/column shape (index=line items, columns=dates) ----------
# yahooquery returns DataFrame: rows = periods, columns = asOfDate, TotalRevenue, NetIncome, ...
# Use tuples for alternate column names so F-Score and Radar get correct values.
_INCOME_ROW_MAP = [
    ("Total Revenue", ("TotalRevenue", "OperatingRevenue", "TotalRevenue")),
    ("Cost Of Revenue", ("CostOfRevenue", "ReconciledCostOfRevenue")),
    ("Gross Profit", ("GrossProfit",)),
    ("Operating Income", ("OperatingIncome", "EBIT", "TotalOperatingIncomeAsReported")),
    ("Net Income", ("NetIncome", "NetIncomeCommonStockholders", "NetIncomeContinuousOperations", "DilutedNIAvailtoComStockholders")),
    ("Operating Expense", ("OperatingExpense", "OperatingExpenses", "TotalExpenses")),
    ("Interest Expense", ("InterestExpense", "InterestExpenseNonOperating")),
    ("Research And Development Expenses", ("ResearchAndDevelopment", "ResearchAndDevelopmentExpenses")),
]
_BALANCE_ROW_MAP = [
    ("Total Assets", ("TotalAssets",)),
    ("Total Stockholder Equity", ("StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest")),
    ("Total Liabilities", ("TotalLiabilitiesNetMinorityInterest", "TotalLiabilities")),
    ("Current Assets", ("CurrentAssets",)),
    ("Current Liabilities", ("CurrentLiabilities",)),
    ("Long Term Debt", ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligation")),
    ("Total Debt", ("TotalDebt",)),
    ("Share Issued", ("OrdinarySharesNumber", "ShareIssued", "BasicAverageShares", "DilutedAverageShares")),
    ("Cash And Cash Equivalents", ("CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "EndCashPosition")),
    ("Retained Earnings", ("RetainedEarnings",)),
]
_CASHFLOW_ROW_MAP = [
    ("Operating Cash Flow", ("OperatingCashFlow", "CashFromOperatingActivities")),
    ("Capital Expenditure", ("CapitalExpenditure", "CapitalExpenditures")),
]


def _yq_df_to_our_shape(df: pd.DataFrame, row_map: list, date_col: str = "asOfDate") -> Optional[pd.DataFrame]:
    """Convert yahooquery DataFrame (rows=periods, columns=line items) to our shape: index=line names, columns=dates."""
    if df is None or df.empty or date_col not in df.columns:
        return None
    df = df.dropna(subset=[date_col]).sort_values(date_col, ascending=False).head(5)
    if df.empty:
        return None
    dates = df[date_col].astype(str).str[:10].tolist()
    data = {}
    for our_name, yq_col in row_map:
        cols = (yq_col,) if isinstance(yq_col, str) else yq_col
        val_col = next((c for c in cols if c in df.columns), None)
        if val_col is None:
            data[our_name] = [None] * len(dates)
            continue
        data[our_name] = [_safe_float(v) for v in df[val_col].tolist()]
    out = pd.DataFrame(data, index=dates).T
    out.columns = dates
    return out


def _share_issued_from_yq_balance(df_bal: pd.DataFrame) -> Optional[pd.Series]:
    """Try OrdinarySharesNumber then ShareIssued for shares outstanding in yahooquery balance."""
    if df_bal is None or df_bal.empty:
        return None
    for col in ("OrdinarySharesNumber", "ShareIssued"):
        if col in df_bal.columns and "asOfDate" in df_bal.columns:
            s = df_bal.set_index("asOfDate")[col].sort_index(ascending=False)
            s.index = s.index.astype(str).str[:10]
            return s.reindex(s.index)  # keep as series with date index
    return None


@st.cache_data(ttl=300)
def _get_annual_financials_balance_cashflow_yahooquery(ticker: str) -> tuple:
    """Fetch income, balance, cash flow from yahooquery. Return (fin_df, bal_df, cf_df) with index=line items, columns=dates. TTM fallback if annual insufficient."""
    if not YQTicker or not ticker:
        return (None, None, None)
    try:
        yq = YQTicker(ticker.upper())
        inc_a = yq.income_statement(frequency="a", trailing=False)
        bal_a = yq.balance_sheet(frequency="a", trailing=False)
        cf_a = yq.cash_flow(frequency="a", trailing=False)
        if inc_a is None or inc_a.empty or bal_a is None or bal_a.empty:
            inc_q = yq.income_statement(frequency="q", trailing=False)
            bal_q = yq.balance_sheet(frequency="q", trailing=False)
            cf_q = yq.cash_flow(frequency="q", trailing=False)
            # Build TTM: need at least 2 periods for Piotroski/Radar; use last 4Q and previous 4Q when 8+ quarters
            if inc_q is not None and not inc_q.empty and len(inc_q) >= 4:
                ttm0 = inc_q.head(4).sum(numeric_only=True)
                row0 = ttm0.to_dict() if hasattr(ttm0, "to_dict") else dict(ttm0)
                row0["asOfDate"] = inc_q["asOfDate"].iloc[0] if "asOfDate" in inc_q.columns else "TTM0"
                rows_inc = [row0]
                if len(inc_q) >= 8:
                    ttm1 = inc_q.iloc[4:8].sum(numeric_only=True)
                    row1 = ttm1.to_dict() if hasattr(ttm1, "to_dict") else dict(ttm1)
                    row1["asOfDate"] = inc_q["asOfDate"].iloc[4] if "asOfDate" in inc_q.columns else "TTM1"
                    rows_inc.append(row1)
                inc_a = pd.DataFrame(rows_inc)
            if bal_q is not None and not bal_q.empty:
                bal_a = bal_q.head(2) if (bal_a is None or bal_a.empty) else bal_a
            if cf_q is not None and not cf_q.empty and len(cf_q) >= 4 and (cf_a is None or cf_a.empty):
                ttm0_cf = cf_q.head(4).sum(numeric_only=True)
                row0_cf = ttm0_cf.to_dict() if hasattr(ttm0_cf, "to_dict") else dict(ttm0_cf)
                row0_cf["asOfDate"] = cf_q["asOfDate"].iloc[0] if "asOfDate" in cf_q.columns else "TTM0"
                rows_cf = [row0_cf]
                if len(cf_q) >= 8:
                    ttm1_cf = cf_q.iloc[4:8].sum(numeric_only=True)
                    row1_cf = ttm1_cf.to_dict() if hasattr(ttm1_cf, "to_dict") else dict(ttm1_cf)
                    row1_cf["asOfDate"] = cf_q["asOfDate"].iloc[4] if "asOfDate" in cf_q.columns else "TTM1"
                    rows_cf.append(row1_cf)
                cf_a = pd.DataFrame(rows_cf)
        fin_df = _yq_df_to_our_shape(inc_a, _INCOME_ROW_MAP)
        bal_df = _yq_df_to_our_shape(bal_a, _BALANCE_ROW_MAP)
        if bal_df is not None and "Share Issued" not in bal_df.index and bal_a is not None and not bal_a.empty:
            for sh_col in ("OrdinarySharesNumber", "ShareIssued"):
                if sh_col in bal_a.columns:
                    row = {"Share Issued": [_safe_float(bal_a[sh_col].iloc[0])]}
                    if bal_df is not None and not bal_df.empty:
                        d = str(bal_a["asOfDate"].iloc[0])[:10] if "asOfDate" in bal_a.columns else bal_df.columns[0]
                        extra = pd.DataFrame(row, index=[d]).T
                        extra.columns = [d]
                        bal_df = pd.concat([bal_df, extra], axis=0)
                    break
        cf_df = _yq_df_to_our_shape(cf_a, _CASHFLOW_ROW_MAP)
        return (fin_df, bal_df, cf_df)
    except Exception:
        return (None, None, None)


# ---------- Raw statements & FCF = OCF - CapEx ----------
def _get_row_series(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in names:
        try:
            if name in df.index:
                return df.loc[name].copy()
        except (KeyError, TypeError):
            continue
    return None


def _fin_or_bal_empty(df) -> bool:
    """True if DataFrame is missing, empty, or has no columns (e.g. yfinance returned empty)."""
    return df is None or df.empty or (hasattr(df, "columns") and len(df.columns) == 0)


@st.cache_data(ttl=300)
def _get_annual_financials_balance_cashflow(ticker: str) -> tuple:
    """Return (fin_df, bal_df, cf_df). Uses yahooquery first; if missing/fail, falls back to yfinance with TTM when needed."""
    if not ticker:
        return (None, None, None)
    fin_df, bal_df, cf_df = _get_annual_financials_balance_cashflow_yahooquery(ticker)
    if fin_df is not None and not fin_df.empty and bal_df is not None and not bal_df.empty:
        return (fin_df, bal_df, cf_df)
    if not yf:
        return (None, None, None)
    try:
        t = yf.Ticker(ticker.upper())
        fin = getattr(t, "financials", None)
        bal = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        if _fin_or_bal_empty(fin):
            qf = getattr(t, "quarterly_financials", None)
            if qf is not None and not qf.empty:
                n = len(qf.columns)
                if n >= 8:
                    c0 = qf.iloc[:, :4].sum(axis=1)
                    c1 = qf.iloc[:, 4:8].sum(axis=1)
                    fin = pd.concat([c0, c1], axis=1)
                    fin.columns = ["TTM0", "TTM1"]
                elif n >= 5:
                    c0 = qf.iloc[:, :4].sum(axis=1)
                    c1 = qf.iloc[:, 4:n].sum(axis=1)
                    fin = pd.concat([c0, c1], axis=1)
                    fin.columns = ["TTM0", "TTM1"]
                else:
                    fin = qf.iloc[:, : min(4, n)].sum(axis=1).to_frame("TTM0")
        if _fin_or_bal_empty(bal):
            qb = getattr(t, "quarterly_balance_sheet", None)
            if qb is not None and not qb.empty:
                n = len(qb.columns)
                bal = qb.iloc[:, : min(2, n)].copy()
                if bal.shape[1] == 1:
                    bal.columns = ["B0"]
                else:
                    bal.columns = ["B0", "B1"]
        if _fin_or_bal_empty(cf):
            qc = getattr(t, "quarterly_cashflow", None)
            if qc is not None and not qc.empty:
                n = len(qc.columns)
                cf = qc.iloc[:, : min(4, n)].sum(axis=1).to_frame("TTM0")
        return (fin, bal, cf)
    except Exception:
        return (None, None, None)


@st.cache_data(ttl=300)
def get_sector_industry(ticker: str) -> dict:
    """Return sector and industry from yfinance. Fallback to N/A."""
    if not yf:
        return {"sector": "N/A", "industry": "N/A"}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        sector = (info.get("sector") or info.get("sectorDisp") or "N/A").strip() or "N/A"
        industry = (info.get("industry") or info.get("industryDisp") or "N/A").strip() or "N/A"
        return {"sector": sector, "industry": industry}
    except Exception:
        return {"sector": "N/A", "industry": "N/A"}


@st.cache_data(ttl=300)
def get_5yr_financial_trend(ticker: str) -> pd.DataFrame:
    """Extract up to 5 years: Revenue, Net Income, Operating Margin, FCF (OCF - CapEx). Handles missing years."""
    if not yf:
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker.upper())
        financials = t.financials  # annual
        cashflow = t.cashflow
        if financials is None or financials.empty or cashflow is None or cashflow.empty:
            return pd.DataFrame()
        dates = sorted(financials.columns.tolist(), reverse=True)[:5]
        ocf = _get_row_series(cashflow, "Operating Cash Flow", "Cash From Operating Activities", "Cash From Operations")
        capx = _get_row_series(cashflow, "Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment")
        revenue = _get_row_series(financials, "Total Revenue", "Revenue", "Net Revenue")
        ni = _get_row_series(financials, "Net Income", "Net Income Common Stockholders")
        op_income = _get_row_series(financials, "Operating Income", "EBIT")
        rows = []
        cashflow_cols = list(cashflow.columns) if cashflow is not None else []
        for d in dates:
            yr = d.year if hasattr(d, "year") else int(str(d)[:4])
            rev = _safe_float(revenue.get(d)) if revenue is not None and d in revenue.index else None
            net_i = _safe_float(ni.get(d)) if ni is not None and d in ni.index else None
            op_i = _safe_float(op_income.get(d)) if op_income is not None and d in op_income.index else None
            oper_margin = (op_i / rev * 100) if (op_i is not None and rev and rev != 0) else ((net_i / rev * 100) if (net_i is not None and rev and rev != 0) else None)
            ocf_val = _safe_float(ocf.get(d)) if ocf is not None and d in ocf.index else None
            if ocf_val is None and ocf is not None and cashflow_cols:
                for c in cashflow_cols:
                    if (getattr(c, "year", None) or int(str(c)[:4])) == yr:
                        ocf_val = _safe_float(ocf.get(c))
                        break
            capx_val = _safe_float(capx.get(d)) if capx is not None and d in capx.index else None
            if capx_val is None and capx is not None and cashflow_cols:
                for c in cashflow_cols:
                    if (getattr(c, "year", None) or int(str(c)[:4])) == yr:
                        capx_val = _safe_float(capx.get(c))
                        break
            if ocf_val is not None and capx_val is not None:
                fcf = ocf_val - capx_val
            elif ocf_val is not None:
                fcf = ocf_val
            else:
                fcf = None
            rows.append({
                "Year": yr,
                "Revenue": rev,
                "Net Income": net_i,
                "Operating Margin %": round(oper_margin, 2) if oper_margin is not None else None,
                "FCF": fcf,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _format_shares_display(shares: float) -> str:
    """Format share count for UI, e.g. 15.42B Shares or 1.2B Shares."""
    if shares is None or shares <= 0:
        return "N/A"
    s = float(shares)
    if s >= 1e9:
        return f"{s / 1e9:.2f}B Shares"
    if s >= 1e6:
        return f"{s / 1e6:.2f}M Shares"
    if s >= 1e3:
        return f"{s / 1e3:.2f}K Shares"
    return f"{s:.0f} Shares"


@st.cache_data(ttl=300)
def get_dcf_inputs(ticker: str) -> dict:
    """FCF, Cash, Total Debt, Shares: from yahooquery (via _get_annual_financials) or yfinance fallback."""
    out = {"fcf": None, "total_debt": 0.0, "cash": 0.0, "shares": None}
    if not ticker:
        return out
    try:
        fin, bal, cf = _get_annual_financials_balance_cashflow(ticker)
        if bal is not None and not bal.empty and cf is not None and not cf.empty:
            sh = _get_row_series(bal, "Share Issued")
            out["shares"] = _safe_float(sh.iloc[0]) if sh is not None and len(sh) > 0 else None
            td = _get_row_series(bal, "Total Debt")
            out["total_debt"] = float(td.iloc[0] or 0) if td is not None and len(td) > 0 else 0.0
            cash_s = _get_row_series(bal, "Cash And Cash Equivalents")
            out["cash"] = float(cash_s.iloc[0] or 0) if cash_s is not None and len(cash_s) > 0 else 0.0
            ocf = _get_row_series(cf, "Operating Cash Flow")
            capx = _get_row_series(cf, "Capital Expenditure")
            if ocf is not None and len(ocf) > 0:
                ocf_val = _safe_float(ocf.iloc[0])
                capx_val = _safe_float(capx.iloc[0]) if capx is not None and len(capx) > 0 else 0.0
                if ocf_val is not None:
                    out["fcf"] = ocf_val - (capx_val or 0)
            if out.get("fcf") is not None or out.get("shares") is not None:
                return out
    except Exception:
        pass
    if not yf:
        return out
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        fast_info = getattr(t, "fast_info", None)
        cashflow = getattr(t, "cashflow", None)
        if cashflow is None or cashflow.empty:
            cashflow = getattr(t, "quarterly_cashflow", None)
        balance = getattr(t, "balance_sheet", None)
        if balance is None or balance.empty:
            balance = getattr(t, "quarterly_balance_sheet", None)

        # ----- Shares Outstanding: multi-step fallback (no manual by default) -----
        shares = None
        if fast_info is not None:
            try:
                s = getattr(fast_info, "shares", None)
                if s is None and hasattr(fast_info, "get"):
                    s = fast_info.get("shares")
                if s is not None and float(s) > 0:
                    shares = float(s)
            except (TypeError, ValueError, AttributeError):
                pass
        if shares is None:
            for key in ("sharesOutstanding", "Shares Outstanding", "impliedSharesOutstanding", "Float Shares"):
                s = info.get(key)
                if s is not None and float(s) > 0:
                    shares = float(s)
                    break
        if shares is None and balance is not None and not balance.empty:
            try:
                if "Share Issued" in balance.index:
                    shares = _safe_float(balance.loc["Share Issued"].iloc[0])
                if (shares is None or shares <= 0) and "Ordinary Shares Number" in balance.index:
                    shares = _safe_float(balance.loc["Ordinary Shares Number"].iloc[0])
            except (KeyError, TypeError, IndexError):
                pass
        out["shares"] = shares if (shares is not None and shares > 0) else None

        # ----- Total Debt: fast_info → info → balance -----
        total_debt = None
        if fast_info is not None:
            try:
                d = getattr(fast_info, "total_debt", None) or (fast_info.get("total_debt") if hasattr(fast_info, "get") else None)
                if d is not None and float(d) >= 0:
                    total_debt = float(d)
            except (TypeError, ValueError, AttributeError):
                pass
        if total_debt is None:
            total_debt = info.get("Total Debt")
        if total_debt is None and balance is not None and not balance.empty:
            try:
                if "Total Debt" in balance.index:
                    total_debt = _safe_float(balance.loc["Total Debt"].iloc[0])
            except (KeyError, TypeError, IndexError):
                pass
        out["total_debt"] = float(total_debt) if total_debt is not None else 0.0

        # ----- Cash: fast_info → info → balance -----
        cash = None
        if fast_info is not None:
            try:
                c = getattr(fast_info, "cash", None) or (fast_info.get("cash") if hasattr(fast_info, "get") else None)
                if c is not None and float(c) >= 0:
                    cash = float(c)
            except (TypeError, ValueError, AttributeError):
                pass
        if cash is None:
            cash = info.get("Cash And Cash Equivalents") or info.get("Cash")
        if cash is None and balance is not None and not balance.empty:
            try:
                for row in ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"):
                    if row in balance.index:
                        cash = _safe_float(balance.loc[row].iloc[0])
                        if cash is not None:
                            break
            except (KeyError, TypeError, IndexError):
                pass
        out["cash"] = float(cash) if cash is not None else 0.0

        # ----- Base FCF = OCF - CapEx -----
        ocf = _get_row_series(cashflow, "Operating Cash Flow", "Cash From Operating Activities", "Cash From Operations") if cashflow is not None else None
        capx = _get_row_series(cashflow, "Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment") if cashflow is not None else None
        if ocf is not None and len(ocf) > 0:
            latest_date = ocf.index[0]
            ocf_val = _safe_float(ocf.iloc[0])
            capx_val = _safe_float(capx.get(latest_date)) if (capx is not None and hasattr(capx, "index") and latest_date in getattr(capx, "index", [])) else (_safe_float(capx.iloc[0]) if capx is not None and len(capx) > 0 else None)
            if capx_val is None:
                capx_val = 0.0
            if ocf_val is not None:
                latest_fcf = ocf_val - capx_val
                if latest_fcf == latest_fcf and not (isinstance(latest_fcf, float) and pd.isna(latest_fcf)):
                    out["fcf"] = latest_fcf
        return out
    except Exception:
        return out


def dcf_intrinsic_value(fcf: float, wacc: float, terminal_growth: float, fcf_growth: float, years: int = 5) -> float:
    """5-year DCF: project FCF with fcf_growth, then terminal value; discount at WACC. Returns enterprise value. Robust: avoids div by zero."""
    if fcf is None or fcf <= 0:
        return 0.0
    if wacc <= terminal_growth or wacc <= 0:
        return 0.0
    pv = 0.0
    fcft = float(fcf)
    for t in range(1, years + 1):
        pv += fcft / ((1 + wacc) ** t)
        fcft *= (1 + fcf_growth)
    terminal_fcf = fcft
    tv = terminal_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += tv / ((1 + wacc) ** years)
    return pv


def dcf_10y_2stage(fcf: float, wacc: float, term_growth: float, fcf_growth: float) -> float:
    """10-Year 2-Stage DCF. Stage 1 (Y1–5): FCF grows at fcf_growth. Stage 2 (Y6–10): growth linearly fades from fcf_growth to term_growth by Y10. TV at Y10; discount all to PV."""
    if fcf is None or fcf <= 0:
        return 0.0
    if wacc <= term_growth or wacc <= 0:
        return 0.0
    pv = 0.0
    fcft = float(fcf)
    for t in range(1, 6):
        pv += fcft / ((1 + wacc) ** t)
        fcft *= (1 + fcf_growth)
    for t in range(6, 11):
        fade = (t - 6) / 4.0
        g_t = fcf_growth + fade * (term_growth - fcf_growth)
        fcft *= (1 + g_t)
        pv += fcft / ((1 + wacc) ** t)
    tv = fcft * (1 + term_growth) / (wacc - term_growth)
    pv += tv / ((1 + wacc) ** 10)
    return pv


def excel_style_dcf(fcf_base: float, wacc: float, term_growth: float, fcf_growth: float, total_debt: float, cash: float, shares: float) -> dict:
    """10Y 2-Stage DCF: EV = PV(FCF Y1–10) + PV(TV); Equity = EV - Debt + Cash; Value per share = Equity / Shares."""
    ev = dcf_10y_2stage(fcf_base, wacc, term_growth, fcf_growth)
    equity = ev - total_debt + cash
    shares_safe = float(shares) if (shares is not None and float(shares) > 0) else None
    value_per_share = (equity / shares_safe) if shares_safe else None
    return {"ev": ev, "equity_value": equity, "value_per_share": value_per_share, "shares": shares_safe}


# Aswath Damodaran sector WACC (approx. 2024/2025 baseline). Used for reference in DCF panel.
DAMODARAN_WACC = {
    "Software": 8.5,
    "Retail": 7.5,
    "Hardware": 9.0,
    "Financials": 8.0,
    "Healthcare": 7.2,
    "Consumer": 7.5,
    "Technology": 8.5,
    "Industrial": 7.8,
    "Energy": 8.2,
    "Utilities": 6.5,
}
DAMODARAN_ERP_PCT = 4.6
DAMODARAN_RF_PCT = 4.2


def _damodaran_wacc_for_sector(sector: str) -> float:
    """Map yfinance sector string to closest Damodaran WACC. Default 8.0%."""
    if not sector:
        return 8.0
    s = (sector or "").lower()
    if "software" in s or "technology" in s or "internet" in s:
        return DAMODARAN_WACC.get("Software", 8.5)
    if "hardware" in s or "semiconductor" in s:
        return DAMODARAN_WACC.get("Hardware", 9.0)
    if "retail" in s or "consumer" in s or "cyclical" in s:
        return DAMODARAN_WACC.get("Retail", 7.5)
    if "financial" in s or "bank" in s or "insurance" in s:
        return DAMODARAN_WACC.get("Financials", 8.0)
    if "health" in s or "pharma" in s:
        return DAMODARAN_WACC.get("Healthcare", 7.2)
    if "industrial" in s:
        return DAMODARAN_WACC.get("Industrial", 7.8)
    if "energy" in s or "oil" in s:
        return DAMODARAN_WACC.get("Energy", 8.2)
    if "utilities" in s:
        return DAMODARAN_WACC.get("Utilities", 6.5)
    return 8.0


@st.cache_data(ttl=300)
def get_analyst_consensus(ticker: str) -> dict:
    """Fetch analyst consensus from yfinance: targetMeanPrice, recommendationKey, revenueGrowth, earningsGrowth. Missing → N/A."""
    out = {"targetMeanPrice": "N/A", "recommendationKey": "N/A", "revenueGrowth": "N/A", "earningsGrowth": "N/A"}
    if not yf or not ticker:
        return out
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        tp = info.get("targetMeanPrice")
        if tp is not None:
            try:
                out["targetMeanPrice"] = f"${float(tp):.2f}"
            except (TypeError, ValueError):
                out["targetMeanPrice"] = str(tp)
        rec = info.get("recommendationKey") or info.get("recommendation")
        if rec is not None:
            out["recommendationKey"] = str(rec)
        rg = info.get("revenueGrowth")
        if rg is not None:
            try:
                out["revenueGrowth"] = f"{float(rg) * 100:.1f}%"
            except (TypeError, ValueError):
                out["revenueGrowth"] = str(rg)
        eg = info.get("earningsGrowth")
        if eg is not None:
            try:
                out["earningsGrowth"] = f"{float(eg) * 100:.1f}%"
            except (TypeError, ValueError):
                out["earningsGrowth"] = str(eg)
        return out
    except Exception:
        return out


@st.cache_data(ttl=300)
def get_dcf_smart_defaults(ticker: str) -> dict:
    """Smart default assumptions: WACC from CAPM (Beta), Terminal Growth = 2.5%, FCF Growth from revenueGrowth/earningsGrowth or 8%."""
    out = {"wacc_pct": 10.0, "term_growth_pct": 2.5, "fcf_growth_pct": 8.0}
    if not yf or not ticker:
        return out
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        beta = info.get("beta")
        if beta is None:
            beta = 1.0
        else:
            try:
                beta = float(beta)
            except (TypeError, ValueError):
                beta = 1.0
        risk_free = 4.0
        market_risk_premium = 5.0
        calculated_wacc = risk_free + (beta * market_risk_premium)
        out["wacc_pct"] = round(min(20.0, max(4.0, calculated_wacc)), 1)
        out["term_growth_pct"] = 2.5
        rev_growth = info.get("revenueGrowth") or info.get("earningsGrowth")
        if rev_growth is not None:
            try:
                g = float(rev_growth)
                out["fcf_growth_pct"] = round(min(30.0, max(-10.0, g * 100)), 1)
            except (TypeError, ValueError):
                pass
        return out
    except Exception:
        return out


# ---------- yfinance: Comps (multiples) ----------
@st.cache_data(ttl=300)
def get_comps_data(tickers: tuple) -> pd.DataFrame:
    """Fetch Forward P/E, EV/EBITDA, P/B using forwardPE, enterpriseToEbitda, priceToBook. Missing → None (display as N/A). Robust per-ticker error handling."""
    if not yf:
        return pd.DataFrame()
    rows = []
    for sym in tickers:
        sym = str(sym).strip().upper()
        if not sym:
            continue
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            forward_pe = info.get("forwardPE") or info.get("Forward PE") or info.get("trailingPE") or info.get("Trailing PE")
            ev_ebitda = info.get("enterpriseToEbitda")
            if ev_ebitda is None:
                ev, ebitda = info.get("enterpriseValue"), info.get("ebitda")
                if ev is not None and ebitda is not None and ebitda != 0:
                    ev_ebitda = ev / ebitda
            pb = info.get("priceToBook") or info.get("Price To Book")
            rows.append({
                "Ticker": sym,
                "Forward P/E": round(float(forward_pe), 2) if forward_pe is not None and _safe_float(forward_pe) is not None else None,
                "EV/EBITDA": round(float(ev_ebitda), 2) if ev_ebitda is not None and _safe_float(ev_ebitda) is not None else None,
                "P/B": round(float(pb), 2) if pb is not None and _safe_float(pb) is not None else None,
            })
        except Exception:
            rows.append({"Ticker": sym, "Forward P/E": None, "EV/EBITDA": None, "P/B": None})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------- DuPont, Altman Z, Red Flags, YoY (2–3 years) ----------
def _na(x):
    """Return N/A for None/NaN, else value (for display)."""
    if x is None or (isinstance(x, float) and (pd.isna(x) or x != x)):
        return "N/A"
    return x


@st.cache_data(ttl=300)
def get_dupont_altman_redflags_yoy(ticker: str) -> dict:
    """Returns DuPont (3-step ROE), Altman Z-Score, red flags, YoY. Uses yahooquery then yfinance with TTM fallback."""
    try:
        fin, bal, _ = _get_annual_financials_balance_cashflow(ticker)
        if fin is None or fin.empty or bal is None or bal.empty:
            return {}
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        # TTM columns: keep order TTM0 (current), TTM1 (prior). Else use date sort (newest first).
        col_list = fin.columns.tolist()
        if col_list and str(col_list[0]).startswith("TTM"):
            dates = col_list[:3]
        else:
            dates = sorted(col_list, reverse=True)[:3]
        if not dates:
            return {}
        rev = _get_row_series(fin, "Total Revenue", "Revenue", "Net Revenue")
        ni = _get_row_series(fin, "Net Income", "Net Income Common Stockholders")
        ebit = _get_row_series(fin, "Operating Income", "EBIT")
        gross = _get_row_series(fin, "Gross Profit")
        interest = _get_row_series(fin, "Interest Expense", "Interest Expense Net")
        total_assets = _get_row_series(bal, "Total Assets")
        total_equity = _get_row_series(bal, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest")
        current_assets = _get_row_series(bal, "Current Assets")
        current_liab = _get_row_series(bal, "Current Liabilities")
        retained = _get_row_series(bal, "Retained Earnings")
        total_liab = _get_row_series(bal, "Total Liabilities")
        market_cap = info.get("marketCap") or info.get("Market Cap")
        def _v(s, d):
            if s is None or d not in s.index:
                return None
            return _safe_float(s.get(d))
        rows = []
        for i, d in enumerate(dates):
            yr = int(str(d)[:4]) if (isinstance(d, str) and str(d)[:4].isdigit()) else (d.year if hasattr(d, "year") else (2024 - i))
            r = _v(rev, d)
            net_i = _v(ni, d)
            ta = _v(total_assets, d)
            te = _v(total_equity, d)
            if ta and ta > 0 and te and te > 0 and r and r != 0:
                npm = (net_i / r * 100) if net_i is not None else None
                at = r / ta if r and ta else None
                em = ta / te if ta and te else None
                roe = (net_i / te * 100) if (net_i and te) else (npm * at * em / 100 if (npm and at and em) else None)
            else:
                npm = at = em = roe = None
            gross_p = _v(gross, d)
            gross_margin = (gross_p / r * 100) if (gross_p and r and r != 0) else None
            op_inc = _v(ebit, d)
            op_margin = (op_inc / r * 100) if (op_inc and r and r != 0) else None
            ca = _v(current_assets, d)
            cl = _v(current_liab, d)
            current_ratio = (ca / cl) if (ca and cl and cl != 0) else None
            int_exp = _v(interest, d)
            if op_inc is not None and int_exp is not None and int_exp != 0:
                _ic = op_inc / int_exp
                interest_cov = round(_ic, 2) if (_ic == _ic and not (isinstance(_ic, float) and (pd.isna(_ic) or _ic != _ic))) else None
            else:
                interest_cov = None  # N/A when Interest Expense is 0 or missing (avoid nan%)
            rows.append({
                "Year": yr,
                "Revenue": r, "Net Income": net_i,
                "NPM %": round(npm, 2) if npm is not None else None,
                "Asset Turnover": round(at, 4) if at is not None else None,
                "Equity Mult.": round(em, 2) if em is not None else None,
                "ROE %": round(roe, 2) if roe is not None else None,
                "Gross Margin %": round(gross_margin, 2) if gross_margin is not None else None,
                "Operating Margin %": round(op_margin, 2) if op_margin is not None else None,
                "Current Ratio": round(current_ratio, 2) if current_ratio is not None else None,
                "Interest Coverage": interest_cov,
            })
        dupont_df = pd.DataFrame(rows)
        yoy = []
        if len(dupont_df) >= 2:
            for col in ["NPM %", "ROE %", "Gross Margin %", "Operating Margin %", "Current Ratio", "Interest Coverage"]:
                if col not in dupont_df.columns:
                    continue
                cur = dupont_df[col].iloc[0]
                prev = dupont_df[col].iloc[1]
                if cur is not None and prev is not None and prev != 0 and not (pd.isna(cur) or pd.isna(prev)):
                    if "Margin" in col or "NPM" in col or "ROE" in col:
                        chg_pp = (cur - prev)  # percentage point change (e.g. 7.0 = 7%)
                        if pd.isna(chg_pp) or chg_pp != chg_pp:
                            continue
                        yoy.append({"Ratio": col, "Latest": cur, "Prior": prev, "YoY (pp)": round(chg_pp, 2), "Comment": f"{'Improved' if chg_pp > 0 else 'Declined'} by {abs(chg_pp):.1f}% YoY"})
                    else:
                        pct = (cur - prev) / abs(prev) * 100
                        if pd.isna(pct) or pct != pct:
                            continue
                        yoy.append({"Ratio": col, "Latest": cur, "Prior": prev, "YoY %": round(pct, 1), "Comment": f"{'Up' if pct > 0 else 'Down'} {abs(round(pct, 1))}% YoY"})
        latest_bal_d = bal.columns[0]
        wc = (_v(current_assets, latest_bal_d) or 0) - (_v(current_liab, latest_bal_d) or 0)
        ta_l = _v(total_assets, latest_bal_d)
        re_l = _v(retained, latest_bal_d)
        tl_l = _v(total_liab, latest_bal_d)
        ebit_l = _v(ebit, fin.columns[0])
        sales_l = _v(rev, fin.columns[0])
        altman_z = None
        if ta_l and ta_l > 0 and market_cap is not None and tl_l and tl_l != 0 and sales_l:
            a = wc / ta_l
            b = (re_l or 0) / ta_l
            c = (ebit_l or 0) / ta_l
            d = market_cap / tl_l
            e = sales_l / ta_l
            altman_z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
        red_flags = []
        if len(dupont_df) > 0:
            row0 = dupont_df.iloc[0]
            cr = row0.get("Current Ratio")
            if cr is not None and cr < 1.0:
                red_flags.append({"metric": "Current Ratio", "value": cr, "threshold": 1.0, "flag": "WARNING", "comment": "Current assets do not cover current liabilities; liquidity risk."})
            ic = row0.get("Interest Coverage")
            if ic is not None and ic < 1.5:
                red_flags.append({"metric": "Interest Coverage", "value": ic, "threshold": 1.5, "flag": "WARNING", "comment": "EBIT barely covers interest; default risk."})
        return {
            "dupont": dupont_df,
            "yoy": yoy,
            "altman_z": round(altman_z, 2) if altman_z is not None else None,
            "red_flags": red_flags,
        }
    except (KeyError, TypeError, ZeroDivisionError, IndexError) as e:
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_quarterly_momentum(ticker: str) -> dict:
    """Last 4 quarters Revenue and Net Income from quarterly_financials; QoQ growth for most recent quarter. Returns {df, qoq_revenue_pct, qoq_ni_pct} or empty."""
    out = {"df": None, "qoq_revenue_pct": None, "qoq_ni_pct": None}
    if not yf or not ticker:
        return out
    try:
        t = yf.Ticker(ticker.upper())
        qfin = getattr(t, "quarterly_financials", None)
        if qfin is None or qfin.empty or len(qfin.columns) < 2:
            return out
        rev = _get_row_series(qfin, "Total Revenue", "Revenue", "Net Revenue")
        ni = _get_row_series(qfin, "Net Income", "Net Income Common Stockholders")
        if rev is None and ni is None:
            return out
        cols = list(qfin.columns)[:4]
        rows = []
        for c in cols:
            try:
                if hasattr(c, "strftime"):
                    q = (c.month - 1) // 3 + 1 if hasattr(c, "month") else 1
                    label = c.strftime("%Y") + f"-Q{q}"
                else:
                    label = str(c)[:12]
            except Exception:
                label = str(c)[:12]
            r_val = _safe_float(rev.loc[c]) if rev is not None and c in rev.index else None
            n_val = _safe_float(ni.loc[c]) if ni is not None and c in ni.index else None
            rows.append({"Quarter": label, "Revenue": r_val, "Net Income": n_val})
        out["df"] = pd.DataFrame(rows)
        if len(rows) >= 2:
            r0, r1 = rows[0].get("Revenue"), rows[1].get("Revenue")
            n0, n1 = rows[0].get("Net Income"), rows[1].get("Net Income")
            if r0 is not None and r1 is not None and r1 != 0:
                out["qoq_revenue_pct"] = round((r0 - r1) / abs(r1) * 100, 1)
            if n0 is not None and n1 is not None and n1 != 0:
                out["qoq_ni_pct"] = round((n0 - n1) / abs(n1) * 100, 1)
        return out
    except Exception:
        return out


@st.cache_data(ttl=300)
def get_quarterly_ratio_changes(ticker: str) -> list:
    """QoQ ratio changes: NPM %, ROE %, Gross Margin %, Operating Margin %, Current Ratio, Interest Coverage. Latest quarter vs previous. Returns list of {Metric, Current, Change, Trend}."""
    out = []
    if not yf or not ticker:
        return out
    try:
        t = yf.Ticker(ticker.upper())
        qf = getattr(t, "quarterly_financials", None)
        qb = getattr(t, "quarterly_balance_sheet", None)
        if qf is None or qf.empty or qb is None or qb.empty or len(qf.columns) < 2 or len(qb.columns) < 2:
            return out
        rev = _get_row_series(qf, "Total Revenue", "Revenue", "Net Revenue")
        ni = _get_row_series(qf, "Net Income", "Net Income Common Stockholders")
        gross = _get_row_series(qf, "Gross Profit")
        ebit = _get_row_series(qf, "Operating Income", "EBIT")
        interest = _get_row_series(qf, "Interest Expense", "Interest Expense Net")
        ta = _get_row_series(qb, "Total Assets")
        te = _get_row_series(qb, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest")
        ca = _get_row_series(qb, "Current Assets")
        cl = _get_row_series(qb, "Current Liabilities")
        def v(s, col):
            if s is None or col not in s.index:
                return None
            return _safe_float(s.get(col))
        c0, c1 = qf.columns[0], qf.columns[1]
        b0, b1 = qb.columns[0], qb.columns[1]
        r0, r1 = v(rev, c0), v(rev, c1)
        n0, n1 = v(ni, c0), v(ni, c1)
        g0, g1 = v(gross, c0), v(gross, c1)
        e0, e1 = v(ebit, c0), v(ebit, c1)
        i0, i1 = v(interest, c0), v(interest, c1)
        ta0, ta1 = v(ta, b0), v(ta, b1)
        te0, te1 = v(te, b0), v(te, b1)
        ca0, ca1 = v(ca, b0), v(ca, b1)
        cl0, cl1 = v(cl, b0), v(cl, b1)
        npm0 = (n0 / r0 * 100) if (n0 is not None and r0 and r0 != 0) else None
        npm1 = (n1 / r1 * 100) if (n1 is not None and r1 and r1 != 0) else None
        roe0 = (n0 / te0 * 100) if (n0 is not None and te0 and te0 != 0) else None
        roe1 = (n1 / te1 * 100) if (n1 is not None and te1 and te1 != 0) else None
        gm0 = (g0 / r0 * 100) if (g0 is not None and r0 and r0 != 0) else None
        gm1 = (g1 / r1 * 100) if (g1 is not None and r1 and r1 != 0) else None
        om0 = (e0 / r0 * 100) if (e0 is not None and r0 and r0 != 0) else None
        om1 = (e1 / r1 * 100) if (e1 is not None and r1 and r1 != 0) else None
        cr0 = (ca0 / cl0) if (ca0 is not None and cl0 and cl0 != 0) else None
        cr1 = (ca1 / cl1) if (ca1 is not None and cl1 and cl1 != 0) else None
        ic0 = (e0 / i0) if (e0 is not None and i0 and i0 != 0) else None
        ic1 = (e1 / i1) if (e1 is not None and i1 and i1 != 0) else None
        def row(metric, cur, prev, is_pct_point=False):
            if cur is None:
                return None
            if prev is None or (is_pct_point and prev != prev):
                return {"Metric": metric, "Current Value": round(cur, 2), "Change": "—", "Trend": "—"}
            if is_pct_point:
                chg = cur - prev
            else:
                chg = ((cur - prev) / abs(prev) * 100) if prev != 0 else 0
            trend = "↑" if chg > 0 else ("↓" if chg < 0 else "—")
            chg_str = f"{chg:+.1f}%" if not is_pct_point else f"{chg:+.1f} pp"
            return {"Metric": metric, "Current Value": round(cur, 2), "Change": chg_str, "Trend": trend}
        for name, cur, prev, is_pp in [
            ("NPM %", npm0, npm1, True), ("ROE %", roe0, roe1, True), ("Gross Margin %", gm0, gm1, True),
            ("Operating Margin %", om0, om1, True), ("Current Ratio", cr0, cr1, False), ("Interest Coverage", ic0, ic1, False),
        ]:
            r = row(name, cur, prev, is_pp)
            if r:
                out.append(r)
        return out
    except Exception:
        return out


@st.cache_data(ttl=300)
def get_income_statement_sankey_data(ticker: str) -> dict:
    """Latest year (or TTM): Revenue, COGS, Gross Profit, OpEx, Operating Income, Tax/Interest/Other, Net Income. Uses yahooquery then yfinance."""
    out = {"revenue": 0, "cogs": 0, "gross_profit": 0, "opex": 0, "operating_income": 0, "tax_interest_other": 0, "net_income": 0}
    fin, _, _ = _get_annual_financials_balance_cashflow(ticker)
    if fin is None or fin.empty:
        return out
    try:
        rev = _get_row_series(fin, "Total Revenue", "Revenue", "Net Revenue")
        cogs = _get_row_series(fin, "Cost Of Revenue", "Cost Of Goods Sold")
        gross = _get_row_series(fin, "Gross Profit")
        op_inc = _get_row_series(fin, "Operating Income", "EBIT")
        ni = _get_row_series(fin, "Net Income", "Net Income Common Stockholders")
        if rev is None or len(rev) == 0:
            return out
        d = rev.index[0]
        revenue = abs(_safe_float(rev.get(d)) or 0)
        cogs_val = abs(_safe_float(cogs.get(d)) if cogs is not None and d in cogs.index else 0) or 0
        gross_val = _safe_float(gross.get(d)) if gross is not None and d in gross.index else None
        if gross_val is None and revenue and cogs_val is not None:
            gross_val = revenue - cogs_val
        elif gross_val is None:
            gross_val = revenue
        gross_val = abs(gross_val) if gross_val is not None else 0
        op_inc_val = _safe_float(op_inc.get(d)) if op_inc is not None and d in op_inc.index else None
        op_inc_val = op_inc_val if op_inc_val is not None else 0
        ni_val = _safe_float(ni.get(d)) if ni is not None and d in ni.index else None
        ni_val = ni_val if ni_val is not None else 0
        opex_val = max(0, gross_val - op_inc_val) if (gross_val >= op_inc_val) else 0
        tax_interest_other = max(0, op_inc_val - ni_val) if (op_inc_val - ni_val) > 0 else abs(min(0, op_inc_val - ni_val))
        out["revenue"] = max(revenue, 1)
        out["cogs"] = min(cogs_val, revenue - 1e-6)
        out["gross_profit"] = gross_val
        out["opex"] = opex_val
        out["operating_income"] = op_inc_val
        out["tax_interest_other"] = tax_interest_other
        out["net_income"] = ni_val
        return out
    except Exception:
        return out


def _build_sankey_figure(data: dict) -> "go.Figure":
    """Sankey: Revenue -> COGS + Gross Profit; Gross Profit -> OpEx + OpInc; OpInc -> Tax/Interest/Other + Net Income. Profit=green, Expense=red/grey."""
    if go is None:
        return None
    rev, cogs, gp, opex, opinc, tax_other, ni = (
        data["revenue"], data["cogs"], data["gross_profit"], data["opex"],
        data["operating_income"], data["tax_interest_other"], data["net_income"],
    )
    if rev <= 0:
        return None
    nodes = ["Total Revenue", "Cost of Revenue", "Gross Profit", "Operating Expenses", "Operating Income", "Tax/Interest/Other", "Net Income"]
    node_colors = ["rgba(0,150,80,0.8)", "rgba(180,60,60,0.7)", "rgba(0,150,80,0.8)", "rgba(120,120,120,0.7)", "rgba(0,150,80,0.8)", "rgba(120,120,120,0.7)", "rgba(0,180,90,0.9)"]
    source = [0, 0, 2, 2, 4, 4]
    target = [1, 2, 3, 4, 5, 6]
    value = [cogs, gp, opex, opinc, tax_other, ni]
    value = [max(0, float(v)) for v in value]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value),
    )])
    fig.update_layout(title="Income Statement Flow (Latest Year)", height=400, margin=dict(t=40, b=20, l=20, r=20), font=dict(size=12))
    return fig


def sankey_data_from_ai(ai_dict: dict) -> dict:
    """Build Sankey input dict from get_sec_financials_llm result (current_yr). Gross Profit = Revenue - CostOfRevenue; Operating Income = Gross Profit - OperatingExpenses."""
    out = {"revenue": 0, "cogs": 0, "gross_profit": 0, "opex": 0, "operating_income": 0, "tax_interest_other": 0, "net_income": 0}
    cur = (ai_dict or {}).get("current_yr") or {}
    revenue = max(0, (cur.get("Revenue") or 0))
    cogs = max(0, min(cur.get("CostOfRevenue") or 0, revenue - 1e-6))
    gross_profit = revenue - cogs
    opex = max(0, cur.get("OperatingExpenses") or 0)
    operating_income = gross_profit - opex
    net_income = cur.get("NetIncome") or 0
    tax_interest_other = max(0, operating_income - net_income) if operating_income > net_income else abs(min(0, operating_income - net_income))
    out["revenue"] = max(revenue, 1)
    out["cogs"] = cogs
    out["gross_profit"] = gross_profit
    out["opex"] = opex
    out["operating_income"] = operating_income
    out["tax_interest_other"] = tax_interest_other
    out["net_income"] = net_income
    return out


def piotroski_from_ai(ai_dict: dict) -> dict:
    """Piotroski F-Score (0-9) from AI-extracted current_yr vs previous_yr. Returns {score, criteria, used_ttm: True}."""
    out = {"score": 0, "criteria": [], "used_ttm": True}
    cur = (ai_dict or {}).get("current_yr") or {}
    prev = (ai_dict or {}).get("previous_yr") or {}
    if not cur:
        return out
    def v(d, k): return (d.get(k) or 0)
    ni0, ni1 = v(cur, "NetIncome"), v(prev, "NetIncome")
    ocf0 = v(cur, "OperatingCashFlow")
    ta0, ta1 = v(cur, "TotalAssets"), v(prev, "TotalAssets")
    roa0 = (ni0 / ta0 * 100) if ta0 and ta0 != 0 else None
    roa1 = (ni1 / ta1 * 100) if ta1 and ta1 != 0 else None
    c1 = ni0 > 0
    c2 = ocf0 > 0
    c3 = (roa0 is not None and roa1 is not None and roa0 > roa1)
    c4 = ocf0 > ni0
    lt0, lt1 = v(cur, "LongTermDebt"), v(prev, "LongTermDebt")
    c5 = (ta0 and ta1 and (lt0 / ta0) < (lt1 / ta1)) if ta0 and ta1 else False
    ca0, ca1 = v(cur, "CurrentAssets"), v(prev, "CurrentAssets")
    cl0, cl1 = v(cur, "CurrentLiabilities"), v(prev, "CurrentLiabilities")
    cr0 = (ca0 / cl0) if cl0 and cl0 != 0 else None
    cr1 = (ca1 / cl1) if cl1 and cl1 != 0 else None
    c6 = (cr0 is not None and cr1 is not None and cr0 > cr1)
    sh0, sh1 = v(cur, "SharesOutstanding"), v(prev, "SharesOutstanding")
    c7 = (sh0 <= sh1) if (sh0 and sh1) else True
    rev0, rev1 = v(cur, "Revenue"), v(prev, "Revenue")
    gm0 = ((rev0 - v(cur, "CostOfRevenue")) / rev0 * 100) if rev0 and rev0 != 0 else None
    gm1 = ((rev1 - v(prev, "CostOfRevenue")) / rev1 * 100) if rev1 and rev1 != 0 else None
    c8 = (gm0 is not None and gm1 is not None and gm0 > gm1)
    at0 = (rev0 / ta0) if rev0 and ta0 and ta0 != 0 else None
    at1 = (rev1 / ta1) if rev1 and ta1 and ta1 != 0 else None
    c9 = (at0 is not None and at1 is not None and at0 > at1)
    criteria = [
        ("Net Income > 0 (profitability)", c1),
        ("Operating Cash Flow > 0 (cash generative)", c2),
        ("ROA increased vs prior period (improving returns)", c3),
        ("OCF > Net Income (earnings quality, less accruals)", c4),
        ("Leverage decreased: LT Debt/Assets lower (less debt)", c5),
        ("Current Ratio improved (better liquidity)", c6),
        ("No dilution: shares unchanged or lower (no equity raise)", c7),
        ("Gross Margin improved (pricing power)", c8),
        ("Asset Turnover improved (efficiency)", c9),
    ]
    out["score"] = sum(1 for _, p in criteria if p)
    out["criteria"] = criteria
    return out


def radar_metrics_from_ai(ai_dict: dict) -> dict:
    """ROE, Current Ratio, Asset Turnover, Equity Mult, Revenue YoY from AI dict; normalized 0-100 for radar. Equity proxy: TotalAssets - CurrentLiabilities - LongTermDebt."""
    cur = (ai_dict or {}).get("current_yr") or {}
    prev = (ai_dict or {}).get("previous_yr") or {}
    if not cur:
        return {}
    eq0 = (cur.get("TotalAssets") or 0) - (cur.get("CurrentLiabilities") or 0) - (cur.get("LongTermDebt") or 0)
    if eq0 <= 0:
        eq0 = (cur.get("TotalAssets") or 0) * 0.5
    roe = (cur.get("NetIncome") or 0) / eq0 * 100 if eq0 else 0
    ca, cl = cur.get("CurrentAssets") or 0, cur.get("CurrentLiabilities") or 0
    current_ratio = (ca / cl) if cl and cl != 0 else 0
    ta = cur.get("TotalAssets") or 1
    asset_turnover = (cur.get("Revenue") or 0) / ta
    equity_mult = (cur.get("TotalAssets") or 0) / eq0 if eq0 else 0
    rev0, rev1 = cur.get("Revenue") or 0, prev.get("Revenue") or 0
    rev_yoy = ((rev0 - rev1) / rev1 * 100) if rev1 and rev1 != 0 else 0
    return {
        "theta": ["Profitability (ROE)", "Liquidity (Curr.Ratio)", "Efficiency (Asset Turn.)", "Solvency (Equity Mult.)", "Growth (Rev YoY)"],
        "r": _radar_norm(roe, current_ratio, asset_turnover, equity_mult, rev_yoy),
        "labels": ["Profitability (ROE)", "Liquidity (Curr.Ratio)", "Efficiency (Asset Turn.)", "Solvency (Equity Mult.)", "Growth (Rev YoY)"],
    }


@st.cache_data(ttl=300)
def get_radar_metrics_normalized(ticker: str) -> dict:
    """ROE, Current Ratio, Asset Turnover, Equity Mult, Revenue YoY. Normalized to 0-100 for radar. Returns {theta: [...], r: [...], labels: [...]} or empty."""
    if not ticker:
        return {}
    q = get_dupont_altman_redflags_yoy(ticker)
    if not q:
        return {}
    dupont_df = q.get("dupont")
    if dupont_df is None or dupont_df.empty or len(dupont_df) < 2:
        return {}
    row0 = dupont_df.iloc[0]
    row1 = dupont_df.iloc[1]
    roe = row0.get("ROE %") or 0
    cr = row0.get("Current Ratio") or 0
    at = row0.get("Asset Turnover") or 0
    em = row0.get("Equity Mult.") or 0
    rev0 = dupont_df["Revenue"].iloc[0] if "Revenue" in dupont_df.columns else None
    rev1 = dupont_df["Revenue"].iloc[1] if "Revenue" in dupont_df.columns else None
    rev_yoy = ((rev0 - rev1) / rev1 * 100) if (rev0 and rev1 and rev1 != 0) else 0
    def norm_roe(x):
        if x is None: return 50
        return min(100, max(0, (x + 10) / 40 * 100))
    def norm_cr(x):
        if x is None: return 50
        return min(100, max(0, x / 3 * 100))
    def norm_at(x):
        if x is None: return 50
        return min(100, max(0, x * 50))
    def norm_em(x):
        if x is None: return 50
        return min(100, max(0, (x - 0.5) / 2.5 * 100))
    def norm_yoy(x):
        if x is None: return 50
        return min(100, max(0, (x + 20) / 50 * 100))
    return {
        "theta": ["Profitability (ROE)", "Liquidity (Curr.Ratio)", "Efficiency (Asset Turn.)", "Solvency (Equity Mult.)", "Growth (Rev YoY)"],
        "r": [norm_roe(roe), norm_cr(cr), norm_at(at), norm_em(em), norm_yoy(rev_yoy)],
        "labels": ["Profitability (ROE)", "Liquidity (Curr.Ratio)", "Efficiency (Asset Turn.)", "Solvency (Equity Mult.)", "Growth (Rev YoY)"],
    }


def _build_radar_figure(ticker: str) -> "go.Figure":
    """Plotly line_polar radar chart; fill with translucent color."""
    if go is None:
        return None
    data = get_radar_metrics_normalized(ticker)
    if not data or not data.get("r"):
        return None
    theta = data["theta"] + [data["theta"][0]]
    r = data["r"] + [data["r"][0]]
    fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, fill="toself", fillcolor="rgba(30, 120, 200, 0.4)", line=dict(color="rgb(30,120,200)", width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), angularaxis=dict(tickfont=dict(size=11))), title="Financial Health Radar", height=400, showlegend=False)
    return fig


def _build_radar_figure_from_metrics(metrics: dict) -> "go.Figure":
    """Build radar chart from precomputed metrics dict (theta, r, labels). Used for SEC Item 8 AI path."""
    if go is None or not metrics or not metrics.get("r"):
        return None
    theta = metrics["theta"] + [metrics["theta"][0]]
    r = metrics["r"] + [metrics["r"][0]]
    fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, fill="toself", fillcolor="rgba(30, 120, 200, 0.4)", line=dict(color="rgb(30,120,200)", width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), angularaxis=dict(tickfont=dict(size=11))), title="Financial Health Radar (from 10-K Item 8)", height=400, showlegend=False)
    return fig


def _radar_norm(roe_pct, current_ratio, asset_turnover, equity_mult, rev_yoy_pct):
    """Normalize 5 raw metrics to 0–100 for radar (same logic as get_radar_metrics_normalized)."""
    def n_roe(x): return min(100, max(0, (x + 10) / 40 * 100)) if x is not None else 50
    def n_cr(x): return min(100, max(0, x / 3 * 100)) if x is not None else 50
    def n_at(x): return min(100, max(0, x * 50)) if x is not None else 50
    def n_em(x): return min(100, max(0, (x - 0.5) / 2.5 * 100)) if x is not None else 50
    def n_yoy(x): return min(100, max(0, (x + 20) / 50 * 100)) if x is not None else 50
    return [n_roe(roe_pct), n_cr(current_ratio), n_at(asset_turnover), n_em(equity_mult), n_yoy(rev_yoy_pct)]


def _build_radar_from_manual(roe_pct, current_ratio, asset_turnover, equity_mult, rev_yoy_pct) -> "go.Figure":
    """Build radar chart from 5 manually entered ratios (fallback when ticker data missing)."""
    if go is None:
        return None
    theta = ["Profitability (ROE)", "Liquidity (Curr.Ratio)", "Efficiency (Asset Turn.)", "Solvency (Equity Mult.)", "Growth (Rev YoY)"]
    r = _radar_norm(roe_pct, current_ratio, asset_turnover, equity_mult, rev_yoy_pct)
    theta_closed = theta + [theta[0]]
    r_closed = r + [r[0]]
    fig = go.Figure(data=go.Scatterpolar(r=r_closed, theta=theta_closed, fill="toself", fillcolor="rgba(30, 120, 200, 0.4)", line=dict(color="rgb(30,120,200)", width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), angularaxis=dict(tickfont=dict(size=11))), title="Financial Health Radar (Manual)", height=400, showlegend=False)
    return fig


@st.cache_data(ttl=300)
def get_piotroski_fscore(ticker: str) -> dict:
    """Piotroski F-Score (0-9) from last 2 periods. Uses yahooquery then yfinance with TTM fallback. Returns score + criteria + used_ttm."""
    out = {"score": 0, "criteria": [], "used_ttm": False}
    fin, bal, cf = _get_annual_financials_balance_cashflow(ticker)
    if fin is None or fin.empty or bal is None or bal.empty:
        return out
    if cf is None or cf.empty:
        cf = pd.DataFrame()
    try:
        ncol = min(2, len(fin.columns))
        rev = _get_row_series(fin, "Total Revenue", "Revenue")
        ni = _get_row_series(fin, "Net Income", "Net Income Common Stockholders")
        gross = _get_row_series(fin, "Gross Profit")
        ta = _get_row_series(bal, "Total Assets")
        lt_debt = _get_row_series(bal, "Long Term Debt")
        ca = _get_row_series(bal, "Current Assets")
        cl = _get_row_series(bal, "Current Liabilities")
        ocf = _get_row_series(cf, "Operating Cash Flow", "Cash From Operating Activities") if not cf.empty else None
        shares = _get_row_series(bal, "Share Issued") or _get_row_series(bal, "Ordinary Shares Number")
        if shares is None and yf:
            t = yf.Ticker(ticker.upper())
            info = getattr(t, "info", None) or {}
            sh_info = info.get("sharesOutstanding") or info.get("Shares Outstanding")
            if sh_info is not None:
                try:
                    sh_float = float(sh_info)
                    shares = pd.Series([sh_float] * ncol, index=fin.columns[:ncol])
                except (TypeError, ValueError):
                    pass
        def v0(s):
            if s is None or len(s) == 0:
                return None
            x = _safe_float(s.iloc[0])
            return x if (x is not None and x == x and not (isinstance(x, float) and pd.isna(x))) else None
        def v1(s):
            if s is None or len(s) < 2:
                return None
            x = _safe_float(s.iloc[1])
            return x if (x is not None and x == x and not (isinstance(x, float) and pd.isna(x))) else None
        ni0, ni1 = v0(ni), v1(ni)
        ocf0 = v0(ocf) if ocf is not None else None
        ta0, ta1 = v0(ta), v1(ta)
        roa0 = (ni0 / ta0 * 100) if (ni0 is not None and ta0 is not None and ta0 != 0) else None
        roa1 = (ni1 / ta1 * 100) if (ni1 is not None and ta1 is not None and ta1 != 0) else None
        c1 = (ni0 is not None and ni0 > 0)
        c2 = (ocf0 is not None and ocf0 > 0)
        c3 = (roa0 is not None and roa1 is not None and roa0 > roa1)
        c4 = (ocf0 is not None and ni0 is not None and ocf0 > ni0)
        lt0 = v0(lt_debt) or 0
        lt1 = v1(lt_debt) or 0
        c5 = (ta0 is not None and ta0 != 0 and ta1 is not None and ta1 != 0 and (lt0 / ta0) < (lt1 / ta1))
        cl0, cl1 = v0(cl), v1(cl)
        ca0, ca1 = v0(ca), v1(ca)
        cr0 = (ca0 / cl0) if (ca0 is not None and cl0 is not None and cl0 != 0) else None
        cr1 = (ca1 / cl1) if (ca1 is not None and cl1 is not None and cl1 != 0) else None
        c6 = (cr0 is not None and cr1 is not None and cr0 > cr1)
        sh0, sh1 = v0(shares), v1(shares)
        c7 = (sh0 is not None and sh1 is not None and sh0 <= sh1) if (sh0 is not None and sh1 is not None) else True
        rev0, rev1 = v0(rev), v1(rev)
        gm0 = (v0(gross) / rev0 * 100) if (gross is not None and rev0 is not None and rev0 != 0) else None
        gm1 = (v1(gross) / rev1 * 100) if (gross is not None and rev1 is not None and rev1 != 0) else None
        c8 = (gm0 is not None and gm1 is not None and gm0 > gm1)
        at0 = (rev0 / ta0) if (rev0 is not None and ta0 is not None and ta0 != 0) else None
        at1 = (rev1 / ta1) if (rev1 is not None and ta1 is not None and ta1 != 0) else None
        c9 = (at0 is not None and at1 is not None and at0 > at1)
        criteria = [
            ("Net Income > 0 (profitability)", c1),
            ("Operating Cash Flow > 0 (cash generative)", c2),
            ("ROA increased vs prior period (improving returns)", c3),
            ("OCF > Net Income (earnings quality, less accruals)", c4),
            ("Leverage decreased: LT Debt/Assets lower (less debt)", c5),
            ("Current Ratio improved (better liquidity)", c6),
            ("No dilution: shares unchanged or lower (no equity raise)", c7),
            ("Gross Margin improved (pricing power)", c8),
            ("Asset Turnover improved (efficiency)", c9),
        ]
        score = sum(1 for _, p in criteria if p)
        out["score"] = score
        out["criteria"] = criteria
        out["used_ttm"] = bool(fin is not None and hasattr(fin, "columns") and len(fin.columns) > 0 and any(str(c).startswith("TTM") for c in fin.columns))
        return out
    except Exception:
        out["used_ttm"] = False
        return out


@st.cache_data(ttl=300)
def get_sector_specific_metrics(ticker: str, sector: str) -> dict:
    """Technology: Rule of 40, R&D % revenue. Retail/Consumer: Inventory Turnover, Operating Margin. Financials: ROE, ROA."""
    if not yf:
        return {}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        fin = t.financials
        bal = t.balance_sheet
        if fin is None or fin.empty:
            fin = getattr(t, "quarterly_financials", None)
            if fin is not None and not fin.empty:
                fin = fin.iloc[:, :4].sum(axis=1).to_frame()
        if bal is None or bal.empty:
            bal = getattr(t, "quarterly_balance_sheet", None)
        out = {}
        sector_lower = (sector or "").lower()
        if "technology" in sector_lower or "software" in sector_lower or "tech" in sector_lower:
            rev = _get_row_series(fin, "Total Revenue", "Revenue", "Net Revenue")
            ocf = _get_row_series(t.cashflow or getattr(t, "quarterly_cashflow", None), "Operating Cash Flow", "Cash From Operating Activities")
            capx = _get_row_series(t.cashflow or getattr(t, "quarterly_cashflow", None), "Capital Expenditure", "Capital Expenditures")
            rd = _get_row_series(fin, "Research And Development", "Research And Development Expense")
            if rev is not None and len(rev) > 0:
                r0 = _safe_float(rev.iloc[0])
                if ocf is not None and len(ocf) > 0 and capx is not None and len(capx) > 0:
                    fcf = _safe_float(ocf.iloc[0]) - _safe_float(capx.iloc[0])
                    out["FCF Margin %"] = round(fcf / r0 * 100, 2) if r0 and fcf is not None else None
                if rd is not None and len(rd) > 0:
                    out["R&D % of Revenue"] = round(_safe_float(rd.iloc[0]) / r0 * 100, 2) if r0 else None
                rev_growth = None
                if rev is not None and len(rev) >= 2:
                    cur, prev = _safe_float(rev.iloc[0]), _safe_float(rev.iloc[1])
                    if prev and prev != 0:
                        rev_growth = (cur - prev) / prev * 100
                if rev_growth is not None and "FCF Margin %" in out and out["FCF Margin %"] is not None:
                    out["Rule of 40 (Rev Growth + FCF Margin)"] = round(rev_growth + out["FCF Margin %"], 1)
        if "consumer" in sector_lower or "retail" in sector_lower or "cyclical" in sector_lower:
            inv = _get_row_series(bal, "Inventory", "Total Inventory")
            cogs = _get_row_series(fin, "Cost Of Revenue", "Cost Of Goods Sold", "Cost of Goods Sold")
            rev = _get_row_series(fin, "Total Revenue", "Revenue", "Net Revenue")
            op_inc = _get_row_series(fin, "Operating Income", "EBIT")
            if inv is not None and len(inv) > 0 and cogs is not None and len(cogs) > 0:
                inv0 = _safe_float(inv.iloc[0])
                cogs0 = _safe_float(cogs.iloc[0])
                out["Inventory Turnover"] = round(cogs0 / inv0, 2) if inv0 else None
            if rev is not None and len(rev) > 0 and op_inc is not None and len(op_inc) > 0:
                r0 = _safe_float(rev.iloc[0])
                op0 = _safe_float(op_inc.iloc[0])
                out["Operating Margin %"] = round(op0 / r0 * 100, 2) if r0 else None
        if "financial" in sector_lower or "bank" in sector_lower or "insurance" in sector_lower:
            ni = _get_row_series(fin, "Net Income", "Net Income Common Stockholders")
            te = _get_row_series(bal, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest")
            ta = _get_row_series(bal, "Total Assets")
            if ni is not None and te is not None and len(ni) > 0 and len(te) > 0:
                te0 = _safe_float(te.iloc[0])
                ni0 = _safe_float(ni.iloc[0])
                out["ROE %"] = round(ni0 / te0 * 100, 2) if te0 else None
            if ni is not None and ta is not None and len(ni) > 0 and len(ta) > 0:
                ta0 = _safe_float(ta.iloc[0])
                ni0 = _safe_float(ni.iloc[0])
                out["ROA %"] = round(ni0 / ta0 * 100, 2) if ta0 else None
        return out
    except Exception:
        return {}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Professional styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 12px 24px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

st.title("All-in-One Financial Analysis Dashboard")
st.caption("Hybrid: Gemini for qualitative (10-K MD&A & Risks); yahooquery + yfinance for quantitative (DCF, Comps). No API key needed for fundamentals.")

with st.sidebar:
    st.header("Settings")
    _prefs = _load_prefs()
    _default_key = _prefs.get("google_api_key") or os.environ.get("GOOGLE_API_KEY", "")
    _default_email = _prefs.get("sec_email") or os.environ.get("SEC_EDGAR_EMAIL", "")
    google_api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        value=_default_key,
        help="Required for Tab 1 (10-K insights).",
        key="input_google_api_key",
    )
    sec_email = st.text_input(
        "SEC EDGAR Email",
        value=_default_email,
        help="Required for 10-K download.",
        key="input_sec_email",
    )
    remember_me = st.checkbox(
        "Remember API key & email (save locally)",
        value=bool(_prefs),
        help="Store in .app_prefs.json in this project. Uncheck to clear and stop saving.",
        key="remember_me",
    )
    if remember_me and (google_api_key or sec_email):
        _save_prefs(google_api_key, sec_email)
    elif not remember_me and _PREFS_PATH.exists():
        try:
            _PREFS_PATH.unlink()
        except Exception:
            pass
    st.markdown("**🔍 Company search**")
    search_query = st.text_input(
        "Search Company Name (e.g., Apple, 삼성, Mitsubishi)",
        value=st.session_state.get("company_search_input", ""),
        key="company_search_input",
        placeholder="e.g. Apple, 삼성, Mitsubishi",
    )
    if st.button("Search Company", key="search_company_btn"):
        query = (search_query or "").strip()
        if not query:
            st.warning("Enter a company name to search.")
        elif yq_search is None:
            st.warning("yahooquery is not installed; search is unavailable.")
        else:
            try:
                raw_results = yq_search(query)
                if not isinstance(raw_results, dict):
                    raw_results = {}
                quotes = raw_results.get("quotes", []) or []
                skip_types = ("INDEX", "MUTUALFUND")
                quotes = [
                    q for q in quotes
                    if q.get("symbol") and q.get("shortname")
                    and (q.get("quoteType") or "EQUITY") not in skip_types
                ]
                if not quotes:
                    st.session_state["company_search_options"] = []
                    st.session_state["company_search_symbols"] = []
                    st.warning("No valid equities found. Try typing the English name (e.g., 'Samsung' instead of '삼성').")
                else:
                    options = []
                    symbols = []
                    for q in quotes[:50]:
                        sym = (q.get("symbol") or "").strip()
                        options.append(f"[{q.get('exchange', 'N/A')}] {q.get('symbol')} - {q.get('shortname', 'Unknown')}")
                        symbols.append(sym)
                    st.session_state["company_search_options"] = options
                    st.session_state["company_search_symbols"] = symbols
                    st.session_state["ticker"] = symbols[0]
                    st.success(f"Found {len(options)} result(s). Select below.")
            except Exception:
                st.warning("No valid equities found. Try typing the English name (e.g., 'Samsung' instead of '삼성').")
                st.session_state["company_search_options"] = []
                st.session_state["company_search_symbols"] = []

    search_options = st.session_state.get("company_search_options") or []
    search_symbols = st.session_state.get("company_search_symbols") or []
    placeholder = "— Click the search button above —"
    options_for_select = [placeholder] if not search_options else search_options
    current_ticker = st.session_state.get("ticker", "NVDA")
    default_idx = 0
    if search_symbols and current_ticker:
        for i, sym in enumerate(search_symbols):
            if sym == current_ticker:
                default_idx = i
                break
    selected_option = st.selectbox(
        "Select company (ticker - name)",
        options=options_for_select,
        index=0 if not search_options else min(default_idx, len(search_options) - 1),
        key="company_select",
    )
    if search_options and selected_option and selected_option != placeholder and " - " in selected_option:
        first_part = selected_option.split(" - ", 1)[0].strip()
        sym = first_part.split("]", 1)[-1].strip() if "]" in first_part else first_part
        st.session_state["ticker"] = sym
    ticker = st.session_state.get("ticker") or (search_symbols[0] if search_symbols else "NVDA")
    st.session_state["google_api_key"] = google_api_key
    st.session_state["sec_email"] = sec_email
    st.session_state["ticker"] = ticker
    st.session_state["market"] = infer_market_from_ticker(ticker)
    st.caption("Search by name (any language), then select. Ticker suffix is set automatically.")

tab1, tab2, tab3 = st.tabs(["10-K & MD&A Insights", "3-Scenario DCF Valuation", "Industry Analysis & Comps"])

# ----- Tab 1: Qualitative (MD&A) + Quantitative (DuPont, Altman Z, Red Flags, YoY) -----
with tab1:
    market = st.session_state.get("market") or MARKET_OPTIONS[0]
    quant_ticker = get_global_ticker(ticker, market) if ticker else ""
    st.subheader("10-K & MD&A Insights — Qualitative and Quantitative")
    if ticker:
        si = get_sector_industry(quant_ticker)
        sector, industry = si.get("sector", "N/A"), si.get("industry", "N/A")
        st.caption(f"Sector: **{sector}**  ·  Industry: **{industry}**" + (f"  ·  Ticker: **{quant_ticker}**" if quant_ticker != ticker else ""))
    if ticker:
        google_api_key = (st.session_state.get("google_api_key") or "").strip()
        sec_email = (st.session_state.get("sec_email") or "").strip()
        ai_data = {}
        if market and "US" in market and google_api_key and sec_email:
            with st.spinner("SEC 10-K 원본에서 재무제표 데이터를 해독하여 그래프를 생성 중입니다... (약 30~60초 소요)"):
                sections, _ = get_10k_sections(ticker, sec_email)
                item8 = (sections or {}).get("item8") or ""
                if item8.strip():
                    ai_data = get_sec_financials_llm(google_api_key, item8, ticker)
        q = get_dupont_altman_redflags_yoy(quant_ticker)
        dupont_df = (q or {}).get("dupont") if q else None
        if q or ai_data:
            st.markdown("---")
            st.markdown("#### 📊 Financial Health (Tables & Charts)")
            c1, c2 = st.columns(2)
            with c1:
                if ai_data and ai_data.get("current_yr"):
                    sankey_data = sankey_data_from_ai(ai_data)
                else:
                    sankey_data = get_income_statement_sankey_data(quant_ticker)
                if sankey_data.get("revenue", 0) > 0:
                    fig_sankey = _build_sankey_figure(sankey_data)
                    if fig_sankey is not None:
                        st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.caption("Income Statement flow: data not available.")
            with c2:
                if ai_data and ai_data.get("current_yr"):
                    radar_metrics = radar_metrics_from_ai(ai_data)
                    fig_radar = _build_radar_figure_from_metrics(radar_metrics) if radar_metrics else None
                else:
                    fig_radar = _build_radar_figure(quant_ticker)
                if fig_radar is not None:
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.caption("Financial radar: need 2+ years of data.")
                    with st.expander("Manual Data Entry (Radar Chart Fallback)", expanded=False):
                        st.caption("Enter 5 key ratios to plot a custom radar. ROE %, Current Ratio, Asset Turnover, Equity Mult., Revenue YoY %.")
                        roe_man = st.number_input("ROE %", value=15.0, min_value=-50.0, max_value=100.0, step=1.0, key="radar_roe")
                        cr_man = st.number_input("Current Ratio", value=1.5, min_value=0.0, max_value=10.0, step=0.1, key="radar_cr")
                        at_man = st.number_input("Asset Turnover", value=0.8, min_value=0.0, max_value=5.0, step=0.1, key="radar_at")
                        em_man = st.number_input("Equity Mult.", value=2.0, min_value=0.5, max_value=10.0, step=0.1, key="radar_em")
                        yoy_man = st.number_input("Revenue YoY %", value=10.0, min_value=-50.0, max_value=200.0, step=1.0, key="radar_yoy")
                        if st.button("Plot Radar", key="radar_plot_btn"):
                            fig_man = _build_radar_from_manual(roe_man, cr_man, at_man, em_man, yoy_man)
                            if fig_man is not None:
                                st.session_state["radar_manual_fig"] = fig_man
                        if st.session_state.get("radar_manual_fig") is not None:
                            st.plotly_chart(st.session_state["radar_manual_fig"], use_container_width=True)
                    with st.expander("Debug: Raw YahooQuery Data", expanded=False):
                        if YQTicker and quant_ticker:
                            try:
                                yq_ticker = YQTicker(quant_ticker.upper())
                                inc_raw = yq_ticker.income_statement(trailing=False)
                                bal_raw = yq_ticker.balance_sheet(trailing=False)
                                if inc_raw is not None and not inc_raw.empty:
                                    st.caption("Income statement (last 2 periods) — check column names for mapping.")
                                    st.dataframe(inc_raw.tail(2), use_container_width=True, hide_index=True)
                                else:
                                    st.caption("Income statement: no data.")
                                if bal_raw is not None and not bal_raw.empty:
                                    st.caption("Balance sheet (last 2 periods) — check column names for mapping.")
                                    st.dataframe(bal_raw.tail(2), use_container_width=True, hide_index=True)
                                else:
                                    st.caption("Balance sheet: no data.")
                            except Exception as e:
                                st.error(f"YahooQuery debug failed: {e}")
                        else:
                            st.caption("YahooQuery not available or no ticker selected.")
            if ai_data and ai_data.get("current_yr"):
                piot = piotroski_from_ai(ai_data)
            else:
                piot = get_piotroski_fscore(quant_ticker)
            st.markdown("**Piotroski F-Score (9-point checklist)**")
            score = piot.get("score", 0)
            legend = "**Score 8–9: Excellent** · 4–7: Average · 0–3: High Risk"
            st.metric("F-Score", f"{score} / 9", legend)
            if ai_data and ai_data.get("current_yr"):
                st.caption("*(from SEC 10-K Item 8)*")
            elif piot.get("used_ttm"):
                st.caption("*(Estimated via TTM Data)*")
            st.caption("✅ = Good (passes criterion). ❌ = Fails criterion.")
            criteria = piot.get("criteria", [])
            if criteria:
                cols = st.columns(3)
                for i, (label, passed) in enumerate(criteria):
                    with cols[i % 3]:
                        st.caption(("✅ " if passed else "❌ ") + label)
            az = (q or {}).get("altman_z")
            if az is not None:
                st.caption(f"**Altman Z-Score:** {az} (Safe > 2.99 · Grey 1.81–2.99 · Distress < 1.81)")
            red_flags = (q or {}).get("red_flags") or []
            if red_flags:
                for rf in red_flags:
                    val = rf.get("value")
                    val_str = "N/A" if (val is None or (isinstance(val, float) and (pd.isna(val) or val != val))) else val
                    st.warning(f"**{rf.get('metric')}:** {val_str} (threshold: {rf.get('threshold')})")
            elif dupont_df is not None and not dupont_df.empty:
                st.success("No red flags (Current Ratio ≥ 1.0, Interest Coverage ≥ 1.5).")
            sector_metrics = get_sector_specific_metrics(quant_ticker, sector) if quant_ticker else {}
            if sector_metrics:
                st.markdown("**Sector-specific metrics**")
                cols = st.columns(min(len(sector_metrics), 4))
                for i, (k, v) in enumerate(sector_metrics.items()):
                    with cols[i % len(cols)]:
                        disp = f"{v}" if v is not None else "N/A"
                        st.metric(k, disp, None)
            def _style_change_column(df: pd.DataFrame):
                """Green for improvement (+), red for decline (-) in Change column."""
                change_col = "Change (%)" if "Change (%)" in df.columns else "Change"
                if change_col not in df.columns or df.empty:
                    return df.style
                def _cell_style(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return ""
                    s = str(v).strip()
                    if s == "—":
                        return ""
                    if s.startswith("+") or "↑" in s:
                        return "background-color: #d4edda; color: #155724"
                    if s.startswith("-") or "↓" in s:
                        return "background-color: #f8d7da; color: #721c24"
                    return ""
                return df.style.apply(lambda col: [_cell_style(v) for v in col], subset=[change_col])
            yoy_list = (q or {}).get("yoy") or []
            if yoy_list:
                st.markdown("**YoY ratio changes**")
                rows_yoy = []
                for item in yoy_list:
                    cur = item.get("Latest")
                    chg_pp = item.get("YoY (pp)")
                    chg_pct = item.get("YoY %")
                    if chg_pp is not None:
                        chg_str = f"{chg_pp:+.1f}%"
                    elif chg_pct is not None:
                        chg_str = f"{chg_pct:+.1f}%"
                    else:
                        chg_str = "—"
                    status = "↑" if (chg_pp is not None and chg_pp > 0) or (chg_pct is not None and chg_pct > 0) else ("↓" if (chg_pp is not None and chg_pp < 0) or (chg_pct is not None and chg_pct < 0) else "—")
                    rows_yoy.append({"Metric": item.get("Ratio"), "Current Value": cur, "Change (%)": chg_str, "Status": status})
                if rows_yoy:
                    df_yoy = pd.DataFrame(rows_yoy)
                    st.dataframe(_style_change_column(df_yoy), use_container_width=True, hide_index=True)
            st.markdown("**Quarter ratio changes**")
            qmom = get_quarterly_momentum(quant_ticker)
            qoq_rows = get_quarterly_ratio_changes(quant_ticker)
            qoq_r, qoq_n = qmom.get("qoq_revenue_pct"), qmom.get("qoq_ni_pct")
            build = []
            if qoq_r is not None:
                build.append({"Metric": "Revenue", "Current Value": "—", "Change (%)": f"{qoq_r:+.1f}%", "Status": "↑" if qoq_r > 0 else "↓"})
            if qoq_n is not None:
                build.append({"Metric": "Net Income", "Current Value": "—", "Change (%)": f"{qoq_n:+.1f}%", "Status": "↑" if qoq_n > 0 else "↓"})
            for r in qoq_rows:
                r_copy = dict(r)
                if "Change" in r_copy and "Change (%)" not in r_copy:
                    r_copy["Change (%)"] = r_copy.pop("Change", "—")
                if "Trend" in r_copy:
                    r_copy["Status"] = r_copy.pop("Trend", "—")
                build.append(r_copy)
            if build:
                df_q = pd.DataFrame(build)
                if "Change" in df_q.columns and "Change (%)" not in df_q.columns:
                    df_q = df_q.rename(columns={"Change": "Change (%)"})
                if "Trend" in df_q.columns:
                    df_q = df_q.rename(columns={"Trend": "Status"})
                st.dataframe(_style_change_column(df_q), use_container_width=True, hide_index=True)
            elif not qmom.get("df") or qmom["df"].empty:
                st.caption("Quarterly data not available for this ticker.")
        else:
            st.info("Quantitative data not available for this ticker.")
    st.markdown("---")
    st.markdown("#### 🔍 Deep-Dive Analysis (AI)")
    st.caption("10-K sections are cached in **data/**; repeat runs use cache for instant AI analysis. First run may take 20–60 s to fetch 10-K; Gemini then streams in ~5–10 s.")
    if not ticker:
        st.caption("Enter a ticker in the sidebar to enable analysis.")
    else:
        api_ok = bool(st.session_state.get("google_api_key"))
        email_ok = bool(st.session_state.get("sec_email"))
        err_msg = []
        if not api_ok:
            err_msg.append("Google API Key")
        if not email_ok:
            err_msg.append("SEC EDGAR Email")
        if err_msg:
            st.caption(f"Set **{' and '.join(err_msg)}** in the sidebar to run analysis.")
        col_a, col_b = st.columns(2)
        is_us = market and "US" in market
        is_korea = market and ("Korea" in market or "KOSPI" in market or "KOSDAQ" in market)
        is_japan_uk = market and ("Japan" in market or "Nikkei" in market or "UK" in market or "LSE" in market)
        # --- Button A: Management Strategy ---
        with col_a:
            if st.button("Analyze Management Strategy (MD&A)", key="run_mda_strategy"):
                if is_korea:
                    st.warning("DART API integration for Korean MD&A is currently under construction. Please check back in Phase 2.")
                elif is_japan_uk:
                    st.warning("EDINET/LSE document parsing is currently under development.")
                elif not api_ok or not email_ok:
                    st.error("Set API Key and SEC Email in the sidebar.")
                else:
                    try:
                        with st.status("Loading 10-K (cache or download)...", expanded=True) as status:
                            sections, _ = get_10k_sections(ticker, st.session_state["sec_email"])
                            si = get_sector_industry(quant_ticker)
                            status.update(label="10-K loaded. Calling Gemini…", state="running")

                        # Stream OUTSIDE the status box so user sees text as it arrives
                        st.markdown("### Management Strategy (Item 7)")
                        st.caption("Streaming from Gemini (first words in ~5–10 sec, then flows in real time).")
                        stream_gen = get_gemini_item7_strategy_stream(
                            st.session_state["google_api_key"],
                            sections.get("item7") or "",
                            ticker,
                            si.get("sector") or "N/A",
                            si.get("industry") or "N/A",
                        )
                        # write_stream returns the full concatenated string after it finishes streaming
                        full_response = st.write_stream(stream_gen)

                        st.session_state["mda_strategy_result"] = full_response
                        st.session_state["mda_strategy_ticker"] = ticker
                        st.session_state["mda_strategy_error"] = None
                    except Exception as e:
                        st.session_state["mda_strategy_error"] = str(e)
                        st.error(f"Strategy analysis failed: {str(e)}")
        # --- Button B: Risk Factors & Forensic ---
        with col_b:
            if st.button("Analyze Risk Factors (Item 1A)", key="run_mda_risk"):
                if is_korea:
                    st.warning("DART API integration for Korean MD&A is currently under construction. Please check back in Phase 2.")
                elif is_japan_uk:
                    st.warning("EDINET/LSE document parsing is currently under development.")
                elif not api_ok or not email_ok:
                    st.error("Set API Key and SEC Email in the sidebar.")
                else:
                    try:
                        with st.status("Loading 10-K (cache or download)...", expanded=True) as status:
                            sections, _ = get_10k_sections(ticker, st.session_state["sec_email"])
                            status.update(label="10-K loaded. Running forensic audit…", state="running")

                            # Run forensic silently IN THE BACKGROUND first
                            forensic = _gemini_forensic_audit(
                                st.session_state["google_api_key"],
                                sections.get("item3") or "",
                                sections.get("item9a") or "",
                                ticker,
                            )
                            status.update(label="Done.", state="complete")
                        # Stream the risk factors OUTSIDE the status box
                        st.markdown("### Risk Factors (Item 1A)")
                        st.caption("Streaming from Gemini (first words in ~5–10 sec, then flows in real time).")
                        stream_gen = get_gemini_item1a_risks_stream(
                            st.session_state["google_api_key"],
                            sections.get("item1a") or "",
                            ticker,
                        )
                        risk_response = st.write_stream(stream_gen)

                        # Combine both for the final result
                        final_out = risk_response
                        if forensic and forensic.strip():
                            st.markdown("### Forensic Audit (Item 3 & 9A)")
                            st.markdown(forensic.strip())
                            final_out += f"\n\n---\n\n### Forensic Audit (Item 3 & 9A)\n\n{forensic.strip()}"

                        st.session_state["mda_risk_result"] = final_out
                        st.session_state["mda_risk_ticker"] = ticker
                        st.session_state["mda_risk_error"] = None
                    except Exception as e:
                        st.session_state["mda_risk_error"] = str(e)
                        st.error(f"Risk analysis failed: {str(e)}")
        # --- Display Saved Results if User Switches Tabs ---
        st.markdown("---")
        if st.session_state.get("mda_strategy_ticker") == ticker:
            if st.session_state.get("mda_strategy_error"):
                st.error("Strategy Error: " + st.session_state["mda_strategy_error"])
            elif st.session_state.get("mda_strategy_result"):
                with st.expander("View Previous Strategy Analysis", expanded=True):
                    st.markdown(st.session_state["mda_strategy_result"])
        if st.session_state.get("mda_risk_ticker") == ticker:
            if st.session_state.get("mda_risk_error"):
                st.error("Risk Error: " + st.session_state["mda_risk_error"])
            elif st.session_state.get("mda_risk_result"):
                with st.expander("View Previous Risk & Forensic Analysis", expanded=True):
                    st.markdown(st.session_state["mda_risk_result"])

# ----- Tab 2: 5-Year Trend + 3-Scenario DCF -----
with tab2:
    market_t2 = st.session_state.get("market") or MARKET_OPTIONS[0]
    quant_ticker_t2 = get_global_ticker(ticker, market_t2) if ticker else ""
    st.subheader("5-Year Financial Trend & DCF Valuation")
    if ticker:
        si_t2 = get_sector_industry(quant_ticker_t2)
        sector_t2 = (si_t2.get("sector") or "").lower()
        is_financial = "financial" in sector_t2 or "bank" in sector_t2 or "insurance" in sector_t2
    else:
        is_financial = False
    df_trend = get_5yr_financial_trend(quant_ticker_t2) if quant_ticker_t2 else pd.DataFrame()
    if not df_trend.empty and len(df_trend) >= 1:
        st.markdown("#### Key metrics (YoY % change)")
        latest = df_trend.iloc[0]
        prev = df_trend.iloc[1] if len(df_trend) >= 2 else None
        def _yoy_pct(cur, prev_val):
            if prev_val is None or cur is None or prev_val == 0:
                return None
            return (cur - prev_val) / abs(prev_val) * 100
        rev_yoy = _yoy_pct(latest.get("Revenue"), prev.get("Revenue") if prev is not None else None)
        ni_yoy = _yoy_pct(latest.get("Net Income"), prev.get("Net Income") if prev is not None else None)
        om_prev = prev.get("Operating Margin %") if prev is not None else None
        om_cur = latest.get("Operating Margin %")
        om_yoy = (om_cur - om_prev) if (om_cur is not None and om_prev is not None) else None
        fcf_yoy = _yoy_pct(latest.get("FCF"), prev.get("FCF") if prev is not None else None)
        m1, m2, m3, m4 = st.columns(4)
        rev_val = latest.get("Revenue")
        m1.metric("Revenue (latest yr)", f"${rev_val/1e9:.2f}B" if rev_val and rev_val >= 1e9 else (f"${rev_val/1e6:.0f}M" if rev_val else "—"), f"{rev_yoy:+.1f}% YoY" if rev_yoy is not None else None)
        ni_val = latest.get("Net Income")
        m2.metric("Net Income", f"${ni_val/1e9:.2f}B" if ni_val and abs(ni_val) >= 1e9 else (f"${ni_val/1e6:.0f}M" if ni_val is not None else "—"), f"{ni_yoy:+.1f}% YoY" if ni_yoy is not None else None)
        om_val = latest.get("Operating Margin %")
        m3.metric("Operating Margin %", f"{om_val:.1f}%" if om_val is not None else "—", f"{om_yoy:+.1f}pp YoY" if om_yoy is not None else None)
        fcf_val = latest.get("FCF")
        m4.metric("FCF", f"${fcf_val/1e9:.2f}B" if fcf_val and abs(fcf_val) >= 1e9 else (f"${fcf_val/1e6:.0f}M" if fcf_val is not None else "—"), f"{fcf_yoy:+.1f}% YoY" if fcf_yoy is not None else None)
        st.caption("FCF = Operating Cash Flow − Capital Expenditure." + (" For Financials, FCF/EBITDA are less relevant; see ROE/ROA in Tab 1 sector-specific metrics." if is_financial else ""))
        if len(df_trend) >= 2 and px is not None:
            st.markdown("#### 5-year trend: Revenue & FCF")
            df_plot = df_trend.copy()
            df_plot["Revenue_M"] = (df_plot["Revenue"] / 1e6).round(1)
            df_plot["FCF_M"] = (df_plot["FCF"] / 1e6).round(1)
            fig = px.line(df_plot, x="Year", y=["Revenue_M", "FCF_M"], title="Revenue & Free Cash Flow ($M)")
            fig.update_layout(yaxis_title="$M", legend_title="", hovermode="x unified")
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)
    elif ticker:
        st.caption("5-year trend not available for this ticker. DCF section below uses latest FCF from yfinance.")
    st.markdown("---")
    st.markdown("#### DCF valuation (Excel-style): inputs & 3-scenario output")
    dcf_inputs = get_dcf_inputs(quant_ticker_t2) if quant_ticker_t2 else {"fcf": None, "total_debt": 0.0, "cash": 0.0, "shares": None}
    fcf_fetched = dcf_inputs.get("fcf")
    total_debt = float(dcf_inputs.get("total_debt") or 0.0)
    cash = float(dcf_inputs.get("cash") or 0.0)
    shares_fetched = dcf_inputs.get("shares")
    # Base FCF
    if fcf_fetched is None or fcf_fetched <= 0:
        fcf = st.number_input("Base FCF (manual — only if yfinance missing)", value=0.0, min_value=-1e12, step=1e8, format="%.0f", key="dcf_fcf_manual")
    else:
        fcf = float(fcf_fetched)
        st.caption(f"Base FCF (OCF − CapEx): **${fcf/1e9:.2f}B**" if abs(fcf) >= 1e9 else f"Base FCF (OCF − CapEx): **${fcf/1e6:.0f}M**")
    # Shares: auto-fetched (fast_info → info → balance); manual only as last resort
    if shares_fetched is not None and shares_fetched > 0:
        shares = float(shares_fetched)
        st.caption(f"Shares Outstanding: **{_format_shares_display(shares)}** (real-time, auto-fetched)")
    else:
        shares = st.number_input("Shares Outstanding (manual — only if all API sources failed)", value=1e9, min_value=1.0, step=1e7, format="%.0f", key="dcf_shares_manual")
    # Total Debt & Cash: manual only when both API sources completely failed
    if total_debt == 0 and cash == 0:
        c1, c2 = st.columns(2)
        with c1:
            total_debt = st.number_input("Total Debt (manual — only if all sources failed)", value=0.0, min_value=0.0, step=1e8, format="%.0f", key="dcf_debt_manual")
        with c2:
            cash = st.number_input("Cash & Equivalents (manual — only if all sources failed)", value=0.0, min_value=0.0, step=1e8, format="%.0f", key="dcf_cash_manual")
    else:
        st.caption(f"Total Debt: **${total_debt/1e9:.2f}B**" if total_debt >= 1e9 else f"Total Debt: **${total_debt/1e6:.0f}M**" if total_debt >= 1e6 else f"Total Debt: **${total_debt:,.0f}**")
        st.caption(f"Cash & Equivalents: **${cash/1e9:.2f}B**" if cash >= 1e9 else f"Cash & Equivalents: **${cash/1e6:.0f}M**" if cash >= 1e6 else f"Cash & Equivalents: **${cash:,.0f}**")
    dcf_defaults = get_dcf_smart_defaults(quant_ticker_t2) if quant_ticker_t2 else {"wacc_pct": 10.0, "term_growth_pct": 2.5, "fcf_growth_pct": 8.0}
    st.markdown("**Assumptions (sliders)**")
    st.caption("💡 Slider defaults are auto-generated based on the company's Beta (CAPM) and revenue growth estimates.")
    col1, col2, col3 = st.columns(3)
    with col1:
        wacc = st.slider("WACC (Discount Rate) %", 4.0, 20.0, float(dcf_defaults["wacc_pct"]), 0.5, key="dcf_wacc") / 100.0
    with col2:
        term_growth = st.slider("Terminal Growth Rate %", -2.0, 6.0, float(dcf_defaults["term_growth_pct"]), 0.25, key="dcf_term") / 100.0
    with col3:
        base_growth = st.slider("Projected FCF Growth (Stage 1, Y1–5) %", -10.0, 30.0, float(dcf_defaults["fcf_growth_pct"]), 0.5, key="dcf_fcf_growth") / 100.0
    bull_growth = base_growth + 0.02
    bear_growth = base_growth - 0.02
    with st.expander("Reference: Analyst & Macro Assumptions", expanded=False):
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**Analyst consensus (yfinance)**")
            analyst = get_analyst_consensus(quant_ticker_t2) if quant_ticker_t2 else {}
            st.markdown(f"- **Target mean price:** {analyst.get('targetMeanPrice', 'N/A')}")
            st.markdown(f"- **Recommendation:** {analyst.get('recommendationKey', 'N/A')}")
            st.markdown(f"- **Revenue growth est.:** {analyst.get('revenueGrowth', 'N/A')}")
            st.markdown(f"- **Earnings growth est.:** {analyst.get('earningsGrowth', 'N/A')}")
        with right_col:
            st.markdown("**Aswath Damodaran — macro baseline**")
            sector_name = get_sector_industry(ticker).get("sector", "N/A") if ticker else "N/A"
            damodaran_wacc = _damodaran_wacc_for_sector(sector_name) if ticker else 8.0
            st.markdown(f"- **Sector WACC (ref.):** {damodaran_wacc:.1f}% (closest: {sector_name})")
            st.markdown(f"- **US equity risk premium (ERP):** {DAMODARAN_ERP_PCT}%")
            st.markdown(f"- **10Y risk-free rate:** {DAMODARAN_RF_PCT}%")
            st.markdown("[Data & methodology (Damodaran)](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.htm) so users can verify.")
    res_base = excel_style_dcf(fcf, wacc, term_growth, base_growth, total_debt, cash, shares)
    res_bull = excel_style_dcf(fcf, wacc, term_growth, bull_growth, total_debt, cash, shares)
    res_bear = excel_style_dcf(fcf, wacc, term_growth, bear_growth, total_debt, cash, shares)
    price_base = res_base.get("value_per_share") or 0.0
    price_bull = res_bull.get("value_per_share") or 0.0
    price_bear = res_bear.get("value_per_share") or 0.0
    current_price = None
    if quant_ticker_t2 and yf:
        try:
            info = yf.Ticker(quant_ticker_t2.upper()).info or {}
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        except Exception:
            pass
    st.markdown("**Intrinsic value vs current price**")
    if current_price is not None and current_price > 0:
        st.metric("Current price", f"${current_price:.2f}", None)
    st.metric("Base case intrinsic value per share", f"${price_base:.2f}" if price_base else "N/A", f"vs current: {(price_base - current_price):.2f}" if (current_price and price_base) else None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Bull (+2% FCF growth)", f"${price_bull:.2f}" if price_bull else "N/A", f"vs Base: +{(price_bull - price_base):.2f}" if (price_bull and price_base) else None)
    c2.metric("Base", f"${price_base:.2f}" if price_base else "N/A", "—")
    c3.metric("Bear (−2% FCF growth)", f"${price_bear:.2f}" if price_bear else "N/A", f"vs Base: {(price_bear - price_base):.2f}" if (price_bear and price_base) else None)
    df_dcf = pd.DataFrame({
        "Scenario": ["Bull", "Base", "Bear"],
        "FCF Growth %": [f"{bull_growth*100:.1f}", f"{base_growth*100:.1f}", f"{bear_growth*100:.1f}"],
        "Intrinsic Value ($)": [round(price_bull, 2) if price_bull else "N/A", round(price_base, 2) if price_base else "N/A", round(price_bear, 2) if price_bear else "N/A"],
    })
    st.dataframe(df_dcf, use_container_width=True, hide_index=True)

# ----- Tab 3: Top-Down Sector Analysis (Industry Comps + AI Outlook) -----
with tab3:
    st.subheader("Top-Down Sector Analysis")
    st.markdown("Select an **industry** to load peer multiples (Forward P/E, EV/EBITDA, P/B). Green = lowest (undervalued), Red = highest. Optionally generate an **AI Industry Outlook**.")
    sector_options = list(SECTORS.keys())
    selected_industry = st.selectbox("Select industry", sector_options, key="sector_select")
    tickers_list = list(SECTORS.get(selected_industry, []))
    if not tickers_list:
        st.warning("No tickers defined for this industry.")
    else:
        with st.spinner("Fetching market data..."):
            df_comps = get_comps_data(tuple(tickers_list))
        if df_comps.empty:
            st.warning("Could not fetch comps from yfinance. One or more tickers may have failed; try again later.")
        else:
            df_display = df_comps.copy()
            for col in ["Forward P/E", "EV/EBITDA", "P/B"]:
                if col not in df_display.columns:
                    continue
                df_display[col] = df_display[col].apply(
                    lambda x: "N/A" if (x is None or (isinstance(x, float) and pd.isna(x))) else x
                )
            try:
                styled = df_comps.style
                for col in ["Forward P/E", "EV/EBITDA", "P/B"]:
                    if col not in df_comps.columns:
                        continue
                    s = pd.to_numeric(df_comps[col], errors="coerce")
                    valid = s.dropna()
                    if len(valid) < 2:
                        continue
                    lo, hi = valid.min(), valid.max()
                    if lo == hi:
                        continue
                    def color_fn(v, lo_val=lo, hi_val=hi):
                        if pd.isna(v):
                            return ""
                        try:
                            x = float(v)
                        except (TypeError, ValueError):
                            return ""
                        if x <= lo_val:
                            return "background-color: rgba(0, 200, 83, 0.35); color: #0d5c2e"
                        if x >= hi_val:
                            return "background-color: rgba(255, 82, 82, 0.35); color: #b71c1c"
                        return ""
                    styled = styled.map(color_fn, subset=[col])
                styled = styled.format(subset=["Forward P/E", "EV/EBITDA", "P/B"], formatter=lambda x: "N/A" if (pd.isna(x) or x is None) else f"{x:.2f}")
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.caption("Lowest multiple in each column = green (relatively undervalued); highest = red.")

    st.markdown("---")
    st.markdown("#### AI Industry Outlook")
    if st.button("Generate Industry Outlook", key="industry_outlook_btn"):
        if not tickers_list:
            st.error("Select an industry above first.")
        elif not st.session_state.get("google_api_key"):
            st.error("Enter your Google API Key in the sidebar.")
        else:
            try:
                with st.spinner("Generating industry outlook with Gemini..."):
                    report = get_industry_outlook(
                        st.session_state["google_api_key"],
                        selected_industry,
                        tickers_list,
                    )
                st.success("Done.")
                st.markdown(report)
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error("Failed to generate outlook. See details below.")
                with st.expander("Error details"):
                    st.code(repr(e), language="text")

