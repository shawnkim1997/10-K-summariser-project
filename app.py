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
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yfinance as yf
except ImportError:
    yf = None


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


# ---------- yfinance: DCF inputs ----------
@st.cache_data(ttl=300)
def get_dcf_inputs(ticker: str) -> dict:
    """Fetch FCF, Total Debt, Cash, Shares Outstanding for DCF. Returns dict or empty on failure."""
    if not yf:
        return {}
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        cashflow = t.cashflow
        balance = t.balance_sheet
        if cashflow is None or cashflow.empty:
            return {}
        fcf_row = None
        for name in ("Free Cash Flow", "Cash From Operations"):
            if name in cashflow.index:
                fcf_row = cashflow.loc[name]
                break
        if fcf_row is None and len(cashflow.index) > 0:
            fcf_row = cashflow.iloc[0]
        latest_fcf = None
        if fcf_row is not None and len(fcf_row) > 0:
            try:
                latest_fcf = float(fcf_row.iloc[0])
            except (TypeError, ValueError):
                pass
        if latest_fcf is not None and (latest_fcf != latest_fcf or latest_fcf <= 0):
            latest_fcf = None
        total_debt = info.get("Total Debt")
        cash = info.get("Cash And Cash Equivalents") or info.get("Cash")
        shares = info.get("Shares Outstanding") or info.get("Float Shares")
        if balance is not None and not balance.empty:
            if total_debt is None and "Total Debt" in balance.index:
                try:
                    total_debt = float(balance.loc["Total Debt"].iloc[0])
                except (TypeError, ValueError, KeyError):
                    pass
            if cash is None and "Cash And Cash Equivalents" in balance.index:
                try:
                    cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
                except (TypeError, ValueError, KeyError):
                    pass
        return {
            "fcf": latest_fcf,
            "total_debt": total_debt if total_debt is not None else 0,
            "cash": cash if cash is not None else 0,
            "shares": shares if shares is not None and shares > 0 else None,
        }
    except Exception:
        return {}


def dcf_intrinsic_value(fcf: float, wacc: float, terminal_growth: float, revenue_growth: float, years: int = 10) -> float:
    """DCF: project FCF with revenue_growth, terminal value with terminal_growth, discount at WACC. Returns enterprise value."""
    if fcf <= 0 or wacc <= terminal_growth:
        return 0.0
    pv = 0.0
    fcft = fcf
    for t in range(1, years + 1):
        pv += fcft / ((1 + wacc) ** t)
        fcft *= (1 + revenue_growth)
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
    ticker = st.text_input("Primary Ticker", value="NVDA", max_chars=10).strip().upper()
    st.session_state["google_api_key"] = google_api_key
    st.session_state["sec_email"] = sec_email
    st.session_state["ticker"] = ticker

if not ticker:
    ticker = st.session_state.get("ticker", "NVDA") or "NVDA"

tab1, tab2, tab3 = st.tabs(["10-K & MD&A Insights", "3-Scenario DCF Valuation", "Industry Analysis & Comps"])

# ----- Tab 1: 10-K & MD&A Insights -----
with tab1:
    st.subheader("10-K & MD&A Insights (Qualitative)")
    st.markdown("Extract **Item 1A (Risk Factors)** and **Item 7 (MD&A)** from the latest 10-K. Gemini analyses: **Management's Tone**, **Strategic Shifts**, **Hidden Risks**.")
    if st.button("Run 10-K Analysis", key="run_10k"):
        if not ticker:
            st.error("Enter a ticker in the sidebar.")
        elif not st.session_state.get("google_api_key"):
            st.error("Enter your Google API Key in the sidebar.")
        elif not st.session_state.get("sec_email"):
            st.error("Enter your SEC EDGAR email in the sidebar.")
        else:
            try:
                with st.spinner("Downloading 10-K and extracting Item 1A & Item 7..."):
                    full_text, item1a, item7 = download_and_extract_item7_and_1a(ticker, st.session_state["sec_email"])
                with st.spinner("Running Gemini analysis (tone, strategy, risks)..."):
                    analysis = get_mda_insights(
                        st.session_state["google_api_key"], item1a, item7, ticker
                    )
                st.success("Analysis complete.")
                st.markdown(analysis)
                with st.expander("View raw excerpt (Item 1A + Item 7)"):
                    excerpt = (item1a or "") + "\n\n---\n\n" + (item7 or "")
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

# ----- Tab 2: 3-Scenario DCF -----
with tab2:
    st.subheader("3-Scenario DCF Valuation (Quantitative)")
    st.markdown("Uses **yfinance** for FCF, Debt, Cash, Shares. No Gemini. Adjust assumptions with sliders.")
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
            ev_base = dcf_intrinsic_value(fcf, wacc, term_growth, base_growth)
            ev_bull = dcf_intrinsic_value(fcf, wacc, term_growth, bull_growth)
            ev_bear = dcf_intrinsic_value(fcf, wacc, term_growth, bear_growth)
            equity_base = ev_base - total_debt + cash
            equity_bull = ev_bull - total_debt + cash
            equity_bear = ev_bear - total_debt + cash
            price_base = equity_base / shares if shares else 0
            price_bull = equity_bull / shares if shares else 0
            price_bear = equity_bear / shares if shares else 0
            st.markdown("#### Intrinsic Value per Share (3 Scenarios)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bull (+2% growth)", f"${price_bull:.2f}", "Base vs Bull")
            c2.metric("Base", f"${price_base:.2f}", "—")
            c3.metric("Bear (-2% growth)", f"${price_bear:.2f}", "Base vs Bear")
            df_dcf = pd.DataFrame({
                "Scenario": ["Bull", "Base", "Bear"],
                "FCF Growth": [f"{bull_growth*100:.1f}%", f"{base_growth*100:.1f}%", f"{bear_growth*100:.1f}%"],
                "Intrinsic Value ($)": [round(price_bull, 2), round(price_base, 2), round(price_bear, 2)],
            })
            st.dataframe(df_dcf, use_container_width=True, hide_index=True)
        else:
            st.info("FCF or Shares Outstanding not available for this ticker. Try another.")

# ----- Tab 3: Industry Comps -----
with tab3:
    st.subheader("Industry Analysis & Comps")
    st.markdown("Enter **comma-separated competitor tickers** (e.g. `AMD, INTC, QCOM`). Multiples from **yfinance**.")
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
