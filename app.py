"""
All-in-One Financial Analysis Dashboard — Hybrid Architecture
- Tab 1: 10-K & MD&A Insights (Item 7 + Item 1A → Gemini, qualitative only).
- Tab 2: 3-Scenario DCF Valuation (yfinance + sliders, no LLM).
- Tab 3: Industry Comps (yfinance multiples: Forward P/E, EV/EBITDA, P/B).
- Cost-effective: Gemini only for text; all numbers from yfinance.
"""

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

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


@st.cache_data(ttl=300)
def get_dcf_inputs(ticker: str) -> dict:
    """FCF = OCF - CapEx from cashflow; baseline = latest year. Debt, Cash, Shares from balance sheet/info."""
    if not yf:
        return {}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        cashflow = t.cashflow
        balance = t.balance_sheet
        if cashflow is None or cashflow.empty:
            return {}
        ocf = _get_row_series(cashflow, "Operating Cash Flow", "Cash From Operating Activities", "Cash From Operations")
        capx = _get_row_series(cashflow, "Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment")
        if ocf is None or len(ocf) == 0:
            return {}
        latest_date = ocf.index[0]
        ocf_val = _safe_float(ocf.iloc[0])
        capx_val = _safe_float(capx.get(latest_date)) if capx is not None and latest_date in capx.index else (_safe_float(capx.iloc[0]) if capx is not None and len(capx) > 0 else None)
        if capx_val is None:
            capx_val = 0.0
        latest_fcf = (ocf_val - capx_val) if ocf_val is not None else None
        if latest_fcf is not None and (latest_fcf != latest_fcf or latest_fcf <= 0):
            latest_fcf = None
        total_debt = info.get("Total Debt")
        cash = info.get("Cash And Cash Equivalents") or info.get("Cash")
        shares = info.get("Shares Outstanding") or info.get("Float Shares")
        if balance is not None and not balance.empty:
            if total_debt is None and "Total Debt" in balance.index:
                total_debt = _safe_float(balance.loc["Total Debt"].iloc[0])
            if cash is None and "Cash And Cash Equivalents" in balance.index:
                cash = _safe_float(balance.loc["Cash And Cash Equivalents"].iloc[0])
        return {
            "fcf": latest_fcf,
            "total_debt": total_debt if total_debt is not None else 0,
            "cash": cash if cash is not None else 0,
            "shares": shares if shares is not None and shares > 0 else None,
        }
    except Exception:
        return {}


def dcf_intrinsic_value(fcf: float, wacc: float, terminal_growth: float, fcf_growth: float, years: int = 5) -> float:
    """5-year DCF: project FCF with fcf_growth, then terminal value; discount at WACC. Returns enterprise value."""
    if fcf <= 0 or wacc <= terminal_growth:
        return 0.0
    pv = 0.0
    fcft = fcf
    for t in range(1, years + 1):
        pv += fcft / ((1 + wacc) ** t)
        fcft *= (1 + fcf_growth)
    terminal_fcf = fcft
    tv = terminal_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += tv / ((1 + wacc) ** years)
    return pv


# ---------- yfinance: Comps (multiples) ----------
@st.cache_data(ttl=300)
def get_comps_data(tickers: tuple) -> pd.DataFrame:
    """Fetch Forward P/E, EV/EBITDA, P/B for each ticker. Returns styled DataFrame."""
    if not yf:
        return pd.DataFrame()
    rows = []
    for sym in tickers:
        sym = str(sym).strip().upper()
        if not sym:
            continue
        try:
            t = yf.Ticker(sym)
            info = t.info
            forward_pe = info.get("Forward PE") or info.get("Trailing PE")
            pb = info.get("Price To Book")
            ev = info.get("Enterprise Value")
            ebitda = info.get("EBITDA")
            ev_ebitda = (ev / ebitda) if (ev is not None and ebitda is not None and ebitda != 0) else None
            rows.append({
                "Ticker": sym,
                "Forward P/E": round(forward_pe, 2) if forward_pe is not None else None,
                "EV/EBITDA": round(ev_ebitda, 2) if ev_ebitda is not None else None,
                "P/B": round(pb, 2) if pb is not None else None,
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
            interest_cov = (op_inc / int_exp) if (op_inc and int_exp and int_exp != 0) else None
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
                "Interest Coverage": round(interest_cov, 2) if interest_cov is not None else None,
            })
        dupont_df = pd.DataFrame(rows)
        yoy = []
        if len(dupont_df) >= 2:
            for col in ["NPM %", "ROE %", "Gross Margin %", "Operating Margin %", "Current Ratio", "Interest Coverage"]:
                if col not in dupont_df.columns:
                    continue
                cur = dupont_df[col].iloc[0]
                prev = dupont_df[col].iloc[1]
                if cur is not None and prev is not None and prev != 0:
                    if "Margin" in col or "NPM" in col or "ROE" in col:
                        chg_bps = (cur - prev) * 100  # bps for %
                        yoy.append({"Ratio": col, "Latest": cur, "Prior": prev, "YoY (bps)": round(chg_bps, 0), "Comment": f"{'Improved' if chg_bps > 0 else 'Declined'} by {abs(round(chg_bps))} bps YoY"})
                    else:
                        pct = (cur - prev) / abs(prev) * 100
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
    google_api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Required for Tab 1 (10-K insights).",
    )
    sec_email = st.text_input(
        "SEC EDGAR Email",
        value=os.environ.get("SEC_EDGAR_EMAIL", ""),
        help="Required for 10-K download.",
    )
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
                    st.warning(f"**{rf.get('flag', 'WARNING')}** — {rf.get('metric')}: {rf.get('value')} (threshold: {rf.get('threshold')}). {rf.get('comment', '')}")
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
    st.markdown("#### DCF valuation: inputs & 3-scenario output")
    dcf_inputs = get_dcf_inputs(ticker) if ticker else {}
    if not dcf_inputs:
        st.warning("Could not fetch DCF inputs from yfinance. Check ticker or try again.")
    else:
        fcf = dcf_inputs.get("fcf") or 0
        total_debt = dcf_inputs.get("total_debt") or 0
        cash = dcf_inputs.get("cash") or 0
        shares = dcf_inputs.get("shares")
        if fcf and fcf > 0 and shares and shares > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                wacc = st.slider("WACC (%)", 4.0, 20.0, 10.0, 0.5) / 100.0
            with col2:
                term_growth = st.slider("Terminal Growth Rate (%)", -2.0, 6.0, 2.0, 0.25) / 100.0
            with col3:
                base_growth = st.slider("Base Case FCF Growth (%)", -10.0, 30.0, 8.0, 0.5) / 100.0
            bull_growth = base_growth + 0.02
            bear_growth = base_growth - 0.02
            ev_base = dcf_intrinsic_value(fcf, wacc, term_growth, base_growth, years=5)
            ev_bull = dcf_intrinsic_value(fcf, wacc, term_growth, bull_growth, years=5)
            ev_bear = dcf_intrinsic_value(fcf, wacc, term_growth, bear_growth, years=5)
            equity_base = ev_base - total_debt + cash
            equity_bull = ev_bull - total_debt + cash
            equity_bear = ev_bear - total_debt + cash
            price_base = equity_base / shares if shares else 0
            price_bull = equity_bull / shares if shares else 0
            price_bear = equity_bear / shares if shares else 0
            st.markdown("**Intrinsic value per share**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bull (+2% growth)", f"${price_bull:.2f}", f"+{(price_bull - price_base):.2f} vs Base")
            c2.metric("Base", f"${price_base:.2f}", "—")
            c3.metric("Bear (−2% growth)", f"${price_bear:.2f}", f"{(price_bear - price_base):.2f} vs Base")
            df_dcf = pd.DataFrame({
                "Scenario": ["Bull", "Base", "Bear"],
                "FCF Growth": [f"{bull_growth*100:.1f}%", f"{base_growth*100:.1f}%", f"{bear_growth*100:.1f}%"],
                "Intrinsic Value ($)": [round(price_bull, 2), round(price_base, 2), round(price_bear, 2)],
            })
            st.dataframe(df_dcf, use_container_width=True, hide_index=True)
        else:
            st.caption("FCF or Shares not available. FCF = OCF − CapEx from yfinance.")

# ----- Tab 3: Industry Comps (conditional formatting) -----
with tab3:
    st.subheader("Industry Analysis & Comps")
    st.markdown("Enter **comma-separated competitor tickers**. Multiples from **yfinance**. Green = below peer average (undervalued), Red = above (overvalued).")
    comp_tickers = st.text_input("Competitor tickers", value="AMD, INTC, QCOM", key="comps").strip()
    if st.button("Load Comps", key="load_comps"):
        tickers_list = [t.strip().upper() for t in comp_tickers.split(",") if t.strip()]
        if ticker and ticker not in tickers_list:
            tickers_list = [ticker] + tickers_list
        if not tickers_list:
            st.warning("Enter at least one ticker.")
        else:
            df_comps = get_comps_data(tuple(tickers_list))
            if df_comps.empty:
                st.warning("Could not fetch comps from yfinance.")
            else:
                try:
                    styled = df_comps.style
                    for col in ["Forward P/E", "EV/EBITDA", "P/B"]:
                        if col not in df_comps.columns:
                            continue
                        s = pd.to_numeric(df_comps[col], errors="coerce")
                        avg = s.mean()
                        if pd.isna(avg):
                            continue
                        def color_fn(v, avg_val=avg):
                            if pd.isna(v):
                                return ""
                            try:
                                x = float(v)
                            except (TypeError, ValueError):
                                return ""
                            if x < avg_val:
                                return "background-color: rgba(0, 200, 83, 0.3); color: #0d5c2e"
                            if x > avg_val:
                                return "background-color: rgba(255, 82, 82, 0.3); color: #b71c1c"
                            return ""
                        styled = styled.map(lambda v: color_fn(v), subset=[col])
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(df_comps, use_container_width=True, hide_index=True)

st.divider()
with st.expander("S&P 500 sample — Company & Ticker"):
    SP500_SAMPLE = [
        ("NVIDIA Corporation", "NVDA"), ("Apple Inc.", "AAPL"), ("Microsoft Corporation", "MSFT"),
        ("Amazon.com Inc.", "AMZN"), ("Alphabet Inc. (Google)", "GOOGL"), ("Meta Platforms Inc.", "META"),
        ("AMD", "AMD"), ("Intel Corporation", "INTC"), ("Qualcomm Inc.", "QCOM"),
    ]
    df_sp = pd.DataFrame(SP500_SAMPLE, columns=["Company", "Ticker"])
    st.dataframe(df_sp, use_container_width=True, hide_index=True)
