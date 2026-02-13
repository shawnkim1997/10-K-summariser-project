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
import time
from pathlib import Path
from typing import Optional

# Local prefs file for "Remember me" (API key & email). Path is in .gitignore.
_PREFS_PATH = Path(__file__).resolve().parent / ".app_prefs.json"


def _load_prefs() -> dict:
    """Load saved API key and email from local file. Keys: google_api_key, sec_email."""
    try:
        if _PREFS_PATH.exists():
            with open(_PREFS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_prefs(google_api_key: str, sec_email: str) -> None:
    """Save API key and email to local file (only if user opted in)."""
    try:
        with open(_PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump({"google_api_key": (google_api_key or "").strip(), "sec_email": (sec_email or "").strip()}, f, indent=2)
    except Exception:
        pass

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yfinance as yf
except ImportError:
    yf = None

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


def extract_text_from_html(html_path: Path) -> str:
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "lxml")
    except Exception:
        with open(html_path, "r", encoding="latin-1", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


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


def clean_text_for_llm(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"\d+", line) or re.fullmatch(r"[\.\-\s\-]+", line):
            continue
        if re.match(r"^(page\s+\d+|\d+)\s*$", line, re.IGNORECASE) and len(line) < 20:
            continue
        lines.append(line)
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def smart_chunk(section: str, max_chars: int = 20000, head_ratio: float = 0.5) -> str:
    if len(section) <= max_chars:
        return section
    head_size = int(max_chars * head_ratio)
    tail_size = max_chars - head_size - 100
    return section[:head_size] + "\n\n[ ... middle omitted ... ]\n\n" + section[-tail_size:]


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


def download_and_extract_item7_and_1a(ticker: str, email: str) -> tuple[str, str, str]:
    """Fetch 10-K from SEC EDGAR and return full_text, Item 1A (Risk Factors), Item 7 (MD&A)."""
    Downloader = get_edgar_downloader()
    with tempfile.TemporaryDirectory() as tmpdir:
        download_root = Path(tmpdir)
        dl = Downloader("FQDC-10K-Analyzer", email, str(download_root))
        dl.get("10-K", ticker.upper(), limit=1, download_details=True)
        filing_dir = find_downloaded_10k_path(download_root, ticker)
        if not filing_dir:
            raise FileNotFoundError(f"Could not find 10-K for ticker '{ticker}'. Check ticker and SEC EDGAR.")
        full_text = get_main_10k_text(filing_dir)
        if not full_text:
            raise ValueError("Could not extract text from the 10-K.")
    item1a = find_item_section_generic(
        full_text, ITEM1A_PATTERNS, 1, ["Risk", "Factors"], max_chars=80000
    )
    text_after_7 = full_text
    start7 = _find_section_start(full_text, ITEM7_PATTERNS, 7)
    if start7 >= 0:
        text_after_7 = full_text[start7:]
    item7 = find_item_section_generic(
        text_after_7, ITEM7_PATTERNS, 7, ["Management's Discussion", "MD&A", "Analysis"], max_chars=100000
    )
    if not item7 and text_after_7:
        item7 = smart_chunk(text_after_7[:120000], max_chars=20000)
    return full_text, item1a, item7


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
The text below is from the latest 10-K for {ticker}: **Item 1A (Risk Factors)** and **Item 7 (MD&A)**.
Provide a concise report: 1) Management's Tone (Sentiment), 2) Key Strategic Shifts, 3) Major Hidden Risks.{kpi_instruction}
Use clear headings. Under 800 words."""
        full_content = f"""--- 10-K Excerpt ---\n\n{combined_text}\n\n---\n\n{user_prompt}"""
    else:
        latest_clean = smart_chunk(clean_text_for_llm(item7_latest), max_chars=12000)
        past_clean = smart_chunk(clean_text_for_llm(item7_3y_ago), max_chars=12000)
        user_prompt = f"""You are a senior equity analyst. Use British English.
Below are **Item 7 (Management's Discussion and Analysis)** from the 10-K for {ticker}: **LATEST YEAR** and **THREE YEARS AGO**. Perform a **Comparative Analysis**.

1. **Core strategy**: What has changed in the company's stated strategy, priorities, or capital allocation between then and now?
2. **Emerging risks**: What new risks appear in the latest MD&A that were absent or less prominent 3 years ago?
3. **Management's tone**: How has the overall tone (confidence, caution, optimism) shifted? Quote 1–2 phrases from each period if relevant.
4. **Industry-specific KPIs**:{kpi_instruction}

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


# ---------- yfinance: raw statements & FCF = OCF - CapEx ----------
def _safe_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and (x != x or pd.isna(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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
    """FCF = OCF - CapEx. Shares: fast_info.shares → info.sharesOutstanding → impliedSharesOutstanding → balance. Debt/Cash: fast_info → info → balance. Manual input only as last resort."""
    out = {"fcf": None, "total_debt": 0.0, "cash": 0.0, "shares": None}
    if not yf or not ticker:
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
    """Returns DuPont (3-step ROE), Altman Z-Score, red flags, YoY. Uses TTM if annual missing. Handles KeyError/NaN."""
    if not yf:
        return {}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        fin = t.financials
        bal = t.balance_sheet
        if fin is None or fin.empty:
            qfin = getattr(t, "quarterly_financials", None)
            if qfin is not None and not qfin.empty and qfin.shape[1] >= 1:
                fin = qfin.iloc[:, :4].sum(axis=1).to_frame("TTM")
            else:
                return {}
        if bal is None or bal.empty:
            qbal = getattr(t, "quarterly_balance_sheet", None)
            if qbal is not None and not qbal.empty:
                bal = qbal.iloc[:, :1]
            else:
                return {}
        dates = sorted(fin.columns.tolist(), reverse=True)[:3]
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
        for d in dates:
            yr = d.year if hasattr(d, "year") else int(str(d)[:4])
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
                        chg_bps = (cur - prev) * 100  # bps for %
                        if pd.isna(chg_bps) or chg_bps != chg_bps:
                            continue
                        yoy.append({"Ratio": col, "Latest": cur, "Prior": prev, "YoY (bps)": round(chg_bps, 0), "Comment": f"{'Improved' if chg_bps > 0 else 'Declined'} by {abs(round(chg_bps))} bps YoY"})
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
st.caption("Hybrid: Gemini for qualitative (10-K MD&A & Risks); yfinance for quantitative (DCF, Comps). Cost-effective personal research.")

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
    st.markdown("**Ticker / Company search**")
    search_term = st.text_input("Type ticker or company name", value="", key="ticker_search", placeholder="e.g. NVDA or NVIDIA")
    search_upper = (search_term or "").strip().upper()
    search_lower = (search_term or "").strip().lower()
    filtered = [o for o in COMPANY_OPTIONS if search_upper in o.split(" - ")[0] or search_lower in o.lower()] if (search_upper or search_lower) else COMPANY_OPTIONS
    default_idx = 0
    current_ticker = st.session_state.get("ticker", "NVDA")
    for i, o in enumerate(filtered):
        if o.startswith(current_ticker + " - "):
            default_idx = i
            break
    selected = st.selectbox("Select company (ticker - name)", filtered, index=min(default_idx, len(filtered) - 1), key="company_select")
    ticker_from_select = selected.split(" - ")[0].strip() if selected else ""
    manual = st.text_input("Or enter ticker manually", value="", max_chars=10, key="manual_ticker").strip().upper()
    ticker = manual if manual else ticker_from_select
    if not ticker:
        ticker = "NVDA"
    st.session_state["google_api_key"] = google_api_key
    st.session_state["sec_email"] = sec_email
    st.session_state["ticker"] = ticker
    st.caption("Example: type _NVDA_ or _NVIDIA_ then select from list.")

tab1, tab2, tab3 = st.tabs(["10-K & MD&A Insights", "3-Scenario DCF Valuation", "Industry Analysis & Comps"])

# ----- Tab 1: Qualitative (MD&A) + Quantitative (DuPont, Altman Z, Red Flags, YoY) -----
with tab1:
    st.subheader("10-K & MD&A Insights — Qualitative and Quantitative")
    if ticker:
        si = get_sector_industry(ticker)
        sector, industry = si.get("sector", "N/A"), si.get("industry", "N/A")
        st.caption(f"Sector: **{sector}**  ·  Industry: **{industry}**")
    st.markdown("**Quantitative** metrics below (DuPont ROE, Altman Z-Score, Red Flags, YoY trends). **Qualitative** analysis: run comparative MD&A with the button.")
    if ticker:
        q = get_dupont_altman_redflags_yoy(ticker)
        if q:
            st.markdown("#### Quantitative health (2–3 years)")
            dupont_df = q.get("dupont")
            if dupont_df is not None and not dupont_df.empty:
                st.markdown("**3-Step DuPont (ROE = Net Profit Margin × Asset Turnover × Equity Multiplier)**")
                display_cols = [c for c in ["Year", "NPM %", "Asset Turnover", "Equity Mult.", "ROE %"] if c in dupont_df.columns]
                display_dupont = dupont_df[display_cols].copy().rename(columns={"Equity Mult.": "Equity Mult"})
                for col in display_dupont.columns:
                    display_dupont[col] = display_dupont[col].apply(lambda x: "N/A" if (x is None or (isinstance(x, float) and pd.isna(x))) else x)
                st.dataframe(display_dupont, use_container_width=True, hide_index=True)
            az = q.get("altman_z")
            if az is not None:
                st.metric("Altman Z-Score (distress / bankruptcy risk)", f"{az}", "Safe zone > 2.99; Grey 1.81–2.99; Distress < 1.81" if az < 1.81 else ("Grey zone" if az < 2.99 else "Safe zone"))
            red_flags = q.get("red_flags") or []
            if red_flags:
                st.markdown("**Red flags**")
                for rf in red_flags:
                    val = rf.get("value")
                    val_str = "N/A" if (val is None or (isinstance(val, float) and (pd.isna(val) or val != val))) else val
                    st.warning(f"**{rf.get('flag', 'WARNING')}** — {rf.get('metric')}: {val_str} (threshold: {rf.get('threshold')}). {rf.get('comment', '')}")
            elif dupont_df is not None and not dupont_df.empty:
                st.success("No red flags triggered (Current Ratio ≥ 1.0, Interest Coverage ≥ 1.5).")
            sector_metrics = get_sector_specific_metrics(ticker, sector) if ticker else {}
            if sector_metrics:
                st.markdown("**Sector-specific metrics**")
                cols = st.columns(min(len(sector_metrics), 4))
                for i, (k, v) in enumerate(sector_metrics.items()):
                    with cols[i % len(cols)]:
                        disp = f"{v}" if v is not None else "N/A"
                        st.metric(k, disp, None)
            yoy_list = q.get("yoy") or []
            if yoy_list:
                st.markdown("**YoY ratio changes**")
                for item in yoy_list:
                    st.caption(f"**{item.get('Ratio')}**: {item.get('Comment', '')}")
        else:
            st.info("Quantitative data not available for this ticker.")
    st.markdown("---")
    st.markdown("#### Qualitative: MD&A comparative analysis")
    st.markdown("Download **latest 10-K** and **10-K from 3 years ago**. Extract **Item 7 (MD&A)** from both. Gemini: strategy shifts, emerging risks, management tone.")
    if st.button("Run 10-K Comparative Analysis", key="run_10k"):
        if not ticker:
            st.error("Enter a ticker in the sidebar.")
        elif not st.session_state.get("google_api_key"):
            st.error("Enter your Google API Key in the sidebar.")
        elif not st.session_state.get("sec_email"):
            st.error("Enter your SEC EDGAR email in the sidebar.")
        else:
            try:
                with st.spinner("Downloading 10-Ks (latest + 3 years ago) and extracting Item 7..."):
                    item1a, item7_latest, item7_3y_ago, has_comparison = download_item7_latest_and_3y_ago(
                        ticker, st.session_state["sec_email"]
                    )
                if not has_comparison:
                    st.info("Only one or fewer 10-K filings available; showing single-year analysis.")
                with st.spinner("Running Gemini (comparative or single-year analysis)..."):
                    si = get_sector_industry(ticker)
                    analysis = get_mda_comparative_insights(
                        st.session_state["google_api_key"],
                        item1a,
                        item7_latest,
                        item7_3y_ago,
                        ticker,
                        sector=si.get("sector"),
                        industry=si.get("industry"),
                    )
                st.success("Analysis complete.")
                st.markdown(analysis)
                with st.expander("View raw excerpt (Item 1A + Item 7 latest)"):
                    excerpt = (item1a or "") + "\n\n---\n\n" + (item7_latest or "")
                    st.text(excerpt[:12000] + ("..." if len(excerpt) > 12000 else ""))
            except FileNotFoundError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error("An error occurred. See details below.")
                with st.expander("Error details"):
                    st.code(repr(e), language="text")

# ----- Tab 2: 5-Year Trend + 3-Scenario DCF -----
with tab2:
    st.subheader("5-Year Financial Trend & DCF Valuation")
    if ticker:
        si_t2 = get_sector_industry(ticker)
        sector_t2 = (si_t2.get("sector") or "").lower()
        is_financial = "financial" in sector_t2 or "bank" in sector_t2 or "insurance" in sector_t2
    else:
        is_financial = False
    df_trend = get_5yr_financial_trend(ticker) if ticker else pd.DataFrame()
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
    dcf_inputs = get_dcf_inputs(ticker) if ticker else {"fcf": None, "total_debt": 0.0, "cash": 0.0, "shares": None}
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
    dcf_defaults = get_dcf_smart_defaults(ticker) if ticker else {"wacc_pct": 10.0, "term_growth_pct": 2.5, "fcf_growth_pct": 8.0}
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
            analyst = get_analyst_consensus(ticker) if ticker else {}
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
    if ticker and yf:
        try:
            info = yf.Ticker(ticker.upper()).info or {}
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

