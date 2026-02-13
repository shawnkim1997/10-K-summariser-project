"""
10-K Financial Analyzer (Google Gemini) — Hybrid Architecture
- Download 10-K from SEC EDGAR; extract Item 7 (MD&A) only for AI.
- Quantitative: financial metrics (Revenue, Net Income, Operating Cash Flow) from yfinance.
- Qualitative: Item 7 only to Gemini for strategic direction, risks, and sentiment analysis.
- HTML cleansing before sending text to LLM to minimise tokens.
- All content in British English.
"""

import json
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
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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


ITEM7_PATTERNS = [
    r"Item\s+7\s*[.:]\s*Management['\u2019]s\s+Discussion\s+and\s+Analysis",
    r"ITEM\s+7\s*[.:]\s*Management['\u2019]s\s+Discussion",
    r"Item\s+7\s*[.:]\s*[\w\s]+MD&A",
]
ITEM8_PATTERNS = [
    r"Item\s+8\s*[.:]\s*Financial\s+Statements",
    r"ITEM\s+8\s*[.:]\s*Financial\s+Statements",
    r"Item\s+8\s*[.:]\s*[\w\s]+Consolidated\s+Financial",
]


def _find_section_start(text: str, patterns: list, item_num: int) -> int:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.start()
    m = re.search(r"\bItem\s+" + str(item_num) + r"\b", text, re.IGNORECASE)
    return m.start() if m else -1


def prefilter_after_item7(full_text: str) -> str:
    start = _find_section_start(full_text, ITEM7_PATTERNS, 7)
    return full_text[start:] if start >= 0 else full_text


def find_item_section(text: str, item_num: int, title_keywords: list) -> str:
    patterns = ITEM7_PATTERNS if item_num == 7 else ITEM8_PATTERNS
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
    next_item = re.search(r"\n\s*Item\s+\d+\s+", text[start + 100 :], re.IGNORECASE)
    if next_item:
        end = start + 100 + next_item.start()
    else:
        end = min(start + 150000, len(text))
    return text[start:end].strip()


def smart_chunk(section: str, max_chars: int = 30000, head_ratio: float = 0.5) -> str:
    if len(section) <= max_chars:
        return section
    head_size = int(max_chars * head_ratio)
    tail_size = max_chars - head_size - 100
    return (
        section[:head_size]
        + "\n\n[ ... middle omitted to stay within token limit ... ]\n\n"
        + section[-tail_size:]
    )


def clean_text_for_llm(text: str) -> str:
    """
    Token-compression cleansing before sending to LLM: strip HTML remnants,
    collapse whitespace, remove page numbers and excessive special characters.
    """
    if not text or not text.strip():
        return ""
    # Remove any remaining HTML tags (safe on plain text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple spaces to one
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise line endings and collapse many blank lines to at most two newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Drop lines that are only digits (page numbers) or only punctuation/dashes
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"\d+", line) or re.fullmatch(r"[\.\-\s\-]+", line):
            continue
        # Short boilerplate lines (e.g. "Page 1 of 2") — optional: drop very short lines that look like page refs
        if re.match(r"^(page\s+\d+|\d+)\s*$", line, re.IGNORECASE) and len(line) < 20:
            continue
        lines.append(line)
    # Rejoin and collapse again
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


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
    main_path, main_text = max(all_text, key=lambda x: len(x[1]))
    return main_text


GEMINI_MODEL = "gemini-2.0-flash"
RATE_LIMIT_WAIT_SEC = 60
DELAY_BETWEEN_CALLS_SEC = 8


def get_gemini_model(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def _is_rate_limit_error(e: Exception) -> bool:
    err_msg = str(e).lower()
    return (
        "429" in err_msg
        or "resourcelimited" in err_msg
        or "resource exhausted" in err_msg
        or getattr(e, "code", None) == 429
    )


def _generate_with_retry(model, content, generation_config, max_retries: int = 3):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return model.generate_content(content, generation_config=generation_config)
        except Exception as e:
            last_err = e
            if attempt < max_retries and _is_rate_limit_error(e):
                time.sleep(RATE_LIMIT_WAIT_SEC)
                continue
            raise
    raise last_err


def get_metrics_from_yfinance(ticker: str) -> pd.DataFrame:
    """
    Quantitative data: fetch Revenue, Net Income, Operating Cash Flow from yfinance
    (no LLM; fast and accurate). Returns a DataFrame suitable for Streamlit display.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker.upper())
        financials = t.financials  # annual income statement
        cashflow = t.cashflow     # annual cash flow
        if financials is None or financials.empty:
            return pd.DataFrame()
        # Prefer common index names (yfinance varies by region)
        rev_row = None
        for name in ("Total Revenue", "Revenue", "Net Revenue", "Operating Revenue"):
            if name in financials.index:
                rev_row = financials.loc[name]
                break
        ni_row = None
        for name in ("Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"):
            if name in financials.index:
                ni_row = financials.loc[name]
                break
        ocf_row = None
        if cashflow is not None and not cashflow.empty:
            for name in ("Operating Cash Flow", "Cash From Operating Activities", "Cash From Operations"):
                if name in cashflow.index:
                    ocf_row = cashflow.loc[name]
                    break
        # Align by date (columns are often datetime)
        dates = financials.columns.tolist()
        if not dates:
            return pd.DataFrame()
        # Sort descending (most recent first) and take up to 5 years
        dates = sorted(dates, reverse=True)[:5]
        cashflow_cols = list(cashflow.columns) if cashflow is not None and not cashflow.empty else []
        data = {}
        for d in dates:
            yr = d.year if hasattr(d, "year") else int(str(d)[:4])
            rev_val = (rev_row[d] / 1e6) if rev_row is not None and d in rev_row.index else None
            ni_val = (ni_row[d] / 1e6) if ni_row is not None and d in ni_row.index else None
            ocf_val = None
            if ocf_row is not None:
                if d in ocf_row.index:
                    ocf_val = ocf_row[d] / 1e6
                else:
                    for c in cashflow_cols:
                        cy = c.year if hasattr(c, "year") else int(str(c)[:4])
                        if cy == yr:
                            ocf_val = ocf_row[c] / 1e6
                            break
            data[yr] = {"Revenue": rev_val, "Net Income": ni_val, "Operating Cash Flow": ocf_val}
        df = pd.DataFrame(data).T
        df.index.name = "Fiscal Year"
        df = df.astype(float).round(2)
        return df
    except Exception:
        return pd.DataFrame()


def get_ai_summary_and_report(api_key: str, item7_text: str, ticker: str) -> tuple[str, str]:
    """
    Qualitative only: send Item 7 (MD&A) to Gemini. Focus on strategic direction,
    market risks, and sentiment—not on summarising financial statement numbers.
    """
    model = get_gemini_model(api_key)
    item7_text = clean_text_for_llm(item7_text)
    item7_text = smart_chunk(item7_text, max_chars=20000)

    user_prompt = f"""You are a CFA charterholder and senior equity analyst. Use British English.

The text below is Item 7 (Management's Discussion and Analysis) only from the 10-K for company ticker: {ticker}. Do NOT ask for financial statements or numbers—this is a qualitative analysis.

Your task:
1. **Strategic direction**: How does management describe its strategy, priorities, and capital allocation? What are the main growth drivers or initiatives?
2. **Market and business risks**: What material risks (competitive, regulatory, operational, macro) does management emphasise? Be specific and cite the wording where relevant.
3. **Tone (Sentiment)**: Overall, is the tone of MD&A more positive, cautious, or negative? Highlight 2–3 phrases or themes that support your view.

Then write a "CFA INVESTMENT REPORT" section with:
- **Executive Summary**: 2–3 sentences on the company's narrative and management's message.
- **Investment Thesis**: Key strengths and catalysts from the discussion.
- **Key Risks to the Thesis**: Main downside risks from the text.
- **Conclusion**: Balanced wrap-up.

Keep the entire response in British English. Use clear section headers. Do not invent figures—only refer to what is in the text."""

    full_content = f"""--- Item 7. Management's Discussion and Analysis (MD&A) ---\n\n{item7_text}\n\n---\n\n{user_prompt}"""

    try:
        response = _generate_with_retry(model, full_content, {"temperature": 0.3, "max_output_tokens": 8192})
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise

    if not response or not response.text:
        return "No analysis generated.", "No report generated."

    text = response.text.strip()
    detailed, report = text, ""
    if "CFA INVESTMENT REPORT" in text.upper():
        parts = re.split(r"\n\s*(?:CFA INVESTMENT REPORT|CFA Investment Report)\s*\n", text, maxsplit=1, flags=re.IGNORECASE)
        detailed = (parts[0].replace("DETAILED ANALYSIS", "").strip() if parts else "").strip() or text
        report = parts[1].strip() if len(parts) > 1 else ""
    else:
        report = "(CFA Investment Report section not clearly separated; full analysis above.)"

    return detailed, report


def download_and_extract_sections(ticker: str, email: str) -> tuple[str, str, str]:
    Downloader = get_edgar_downloader()
    with tempfile.TemporaryDirectory() as tmpdir:
        download_root = Path(tmpdir)
        dl = Downloader("FQDC-10K-Analyzer", email, str(download_root))
        dl.get("10-K", ticker.upper(), limit=1, download_details=True)
        filing_dir = find_downloaded_10k_path(download_root, ticker)
        if not filing_dir:
            raise FileNotFoundError(f"Could not find 10-K file. Check ticker '{ticker}' and SEC EDGAR response.")
        full_text = get_main_10k_text(filing_dir)
        if not full_text:
            raise ValueError("Could not extract text from the 10-K.")
    text_from_item7 = prefilter_after_item7(full_text)
    item7 = find_item_section(text_from_item7, 7, ["Management's Discussion", "MD&A", "Analysis"])
    item8 = find_item_section(text_from_item7, 8, ["Financial Statements", "Consolidated"])
    if not item7:
        item7 = smart_chunk(text_from_item7[:120000], max_chars=20000)
    if not item8:
        remainder = text_from_item7[100000:220000] if len(text_from_item7) > 100000 else text_from_item7
        item8 = smart_chunk(remainder, max_chars=20000)
    return full_text, item7, item8


def run_analysis(ticker: str, api_key: str, email: str, analysis_only: bool = False) -> tuple[str, str, str, pd.DataFrame]:
    full_text, item7, _ = download_and_extract_sections(ticker, email)
    detailed_summary, cfa_report = get_ai_summary_and_report(api_key, item7, ticker)
    if analysis_only:
        df_metrics = pd.DataFrame()
    else:
        df_metrics = get_metrics_from_yfinance(ticker)
    return detailed_summary, cfa_report, full_text, df_metrics


# ---------- Streamlit UI ----------
st.set_page_config(page_title="10-K Financial Analyzer", layout="wide")
st.title("10-K Financial Analyzer")
st.caption("Hybrid: 10-K Item 7 (MD&A) → Gemini for sentiment & risks; financial metrics from yfinance. British English.")

with st.sidebar:
    st.header("Settings")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password", value=os.environ.get("GOOGLE_API_KEY", ""), help="Obtain from https://aistudio.google.com/apikey")
    email = st.text_input("SEC EDGAR Email Address", value=os.environ.get("SEC_EDGAR_EMAIL", ""), help="Required for SEC programmatic download policy compliance.")
    analysis_only = st.checkbox("Analysis only (1 API call)", value=False, help="Skip metrics table to use only 1 API call.")
    st.session_state["google_api_key"] = google_api_key
    st.session_state["email"] = email
    st.session_state["analysis_only"] = analysis_only

ticker = st.text_input("Stock Ticker (e.g. AAPL, MSFT)", value="AAPL", max_chars=10).strip().upper()
if not ticker:
    st.info("Enter a ticker and click 'Run Analysis', or pick one from the S&P 500 list below.")

SP500_SAMPLE = [
    ("Apple Inc.", "AAPL"), ("Microsoft Corporation", "MSFT"), ("Amazon.com Inc.", "AMZN"),
    ("NVIDIA Corporation", "NVDA"), ("Alphabet Inc. (Google)", "GOOGL"), ("Meta Platforms Inc. (Facebook)", "META"),
    ("Berkshire Hathaway Inc.", "BRK.B"), ("Tesla Inc.", "TSLA"), ("JPMorgan Chase & Co.", "JPM"),
    ("Visa Inc.", "V"), ("UnitedHealth Group Inc.", "UNH"), ("Procter & Gamble Co.", "PG"),
    ("Exxon Mobil Corporation", "XOM"), ("Johnson & Johnson", "JNJ"), ("Mastercard Inc.", "MA"),
    ("Chevron Corporation", "CVX"), ("Home Depot Inc.", "HD"), ("Merck & Co. Inc.", "MRK"),
    ("AbbVie Inc.", "ABBV"), ("Costco Wholesale Corporation", "COST"), ("PepsiCo Inc.", "PEP"),
    ("Coca-Cola Company", "KO"), ("Pfizer Inc.", "PFE"), ("Walmart Inc.", "WMT"), ("Netflix Inc.", "NFLX"),
    ("Adobe Inc.", "ADBE"), ("Salesforce Inc.", "CRM"), ("Comcast Corporation", "CMCSA"), ("Cisco Systems Inc.", "CSCO"),
    ("Oracle Corporation", "ORCL"), ("Intel Corporation", "INTC"), ("American Express Company", "AXP"),
    ("Bank of America Corp.", "BAC"), ("Wells Fargo & Company", "WFC"), ("Verizon Communications Inc.", "VZ"),
    ("AT&T Inc.", "T"), ("Disney (Walt Disney Co.)", "DIS"), ("Nike Inc.", "NKE"), ("McDonald's Corporation", "MCD"),
    ("Starbucks Corporation", "SBUX"), ("Goldman Sachs Group Inc.", "GS"), ("Morgan Stanley", "MS"),
]

st.caption("Select a ticker above or choose from the list below.")

if st.button("Run Analysis"):
    if not ticker:
        st.error("Please enter or select a stock ticker.")
        st.stop()
    api_key = st.session_state.get("google_api_key", "")
    email = st.session_state.get("email", "")
    if not api_key:
        st.error("Please enter your Google API Key (Gemini) in Settings.")
        st.stop()
    if not email:
        st.error("Please enter your SEC EDGAR email address in Settings.")
        st.stop()
    analysis_only = st.session_state.get("analysis_only", False)
    try:
        with st.spinner("Step 1/2: Downloading 10-K and extracting Item 7 (MD&A)..."):
            full_text, item7, _ = download_and_extract_sections(ticker, email)
        with st.spinner("Step 2/2: Running Gemini (qualitative analysis) and fetching financial metrics..."):
            detailed_summary, cfa_report = get_ai_summary_and_report(api_key, item7, ticker)
            if analysis_only:
                df_metrics = pd.DataFrame()
            else:
                df_metrics = get_metrics_from_yfinance(ticker)

        st.success("Analysis complete.")
        st.subheader("Detailed Analysis (Strategy, Risks, Sentiment — from Item 7 MD&A)")
        st.markdown(detailed_summary)
        st.subheader("CFA Investment Report")
        st.markdown(cfa_report)
        st.subheader("Key Financial Metrics (Revenue, Net Income, Operating Cash Flow) — from yfinance")
        if not df_metrics.empty:
            st.dataframe(df_metrics, use_container_width=True)
            st.caption("Values in millions (USD). Source: yfinance.")
        elif analysis_only:
            st.info("Metrics skipped (Analysis only mode).")
        else:
            st.info("No metrics available for this ticker from yfinance.")
        with st.expander("View excerpt of extracted 10-K text"):
            st.text(full_text[:15000] + ("..." if len(full_text) > 15000 else ""))

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))
    except RuntimeError as e:
        st.error(str(e))
        if analysis_only:
            st.warning("You already have Analysis only on. Wait 2–5 minutes, then try again.")
        else:
            st.info("Wait 2–5 minutes, or enable Analysis only (1 API call) in Settings.")
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or ("resource" in err_msg and "exhausted" in err_msg):
            st.error("Rate limit exceeded. Please try again in a few minutes.")
            st.info("Wait 2–5 minutes, or enable **Analysis only (1 API call)** in the sidebar.")
        elif "404" in err_msg or "not found" in err_msg:
            st.error("The selected model is not available. Check Google AI Studio for available models.")
        elif "timeout" in err_msg or "retryerror" in err_msg or "600" in err_msg:
            st.error("Request timed out. The API took too long to respond.")
            st.info("Try again, or enable **Analysis only (1 API call)** to send less data.")
        else:
            st.error("An error occurred. Please try again later.")
            st.caption("If the problem persists, check your API key and internet connection.")
        with st.expander("Error details (for troubleshooting)"):
            st.code(repr(e), language="text")

st.divider()
st.subheader("S&P 500 companies (sample) — Company name & Ticker")
st.caption("Type a ticker from the list into the box above.")
df_sp = pd.DataFrame(SP500_SAMPLE, columns=["Company name", "Ticker"])
with st.expander("Show list", expanded=True):
    st.dataframe(df_sp, use_container_width=True, hide_index=True)
