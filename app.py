"""
10-K Financial Analyzer (Google Gemini 1.5 Flash)
- Download 10-K from SEC EDGAR and extract text
- Analysis using Item 7 (MD&A) and Item 8 (Financial Statements) via Gemini 1.5 Flash (generous free tier, large context)
- CFA-style summary, key metrics table, and CFA Investment Report section
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

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_edgar_downloader():
    from sec_edgar_downloader import Downloader
    return Downloader


def extract_text_from_html(html_path: Path) -> str:
    """Extract plain text from an HTML file."""
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
    """Extract text by file extension (HTML or TXT)."""
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


# ---------- Selective Section Extraction (pre-filter: only Item 7 & 8, no PART I / ITEM 1–6) ----------
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
    """Return start index of first matching pattern, or -1."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.start()
    m = re.search(r"\bItem\s+" + str(item_num) + r"\b", text, re.IGNORECASE)
    return m.start() if m else -1


def prefilter_after_item7(full_text: str) -> str:
    """Drop PART I, ITEM 1–6; keep only from Item 7 onward to reduce noise and token use."""
    start = _find_section_start(full_text, ITEM7_PATTERNS, 7)
    return full_text[start:] if start >= 0 else full_text


def find_item_section(text: str, item_num: int, title_keywords: list) -> str:
    """Extract only Item N section (regex-based). Used for Item 7 (MD&A) and Item 8 (Financial Statements)."""
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
    # End at next "Item N" (next major section)
    next_item = re.search(r"\n\s*Item\s+\d+\s+", text[start + 100 :], re.IGNORECASE)
    if next_item:
        end = start + 100 + next_item.start()
    else:
        end = min(start + 150000, len(text))
    return text[start:end].strip()


def smart_chunk(section: str, max_chars: int = 30000, head_ratio: float = 0.5) -> str:
    """
    If section exceeds max_chars, keep head and tail (quantitative data often at start/end).
    Reduces tokens while preserving high-signal content.
    """
    if len(section) <= max_chars:
        return section
    head_size = int(max_chars * head_ratio)
    tail_size = max_chars - head_size - 100  # reserve for separator
    return (
        section[:head_size]
        + "\n\n[ ... middle omitted to stay within token limit ... ]\n\n"
        + section[-tail_size:]
    )


def find_downloaded_10k_path(download_root: Path, ticker: str) -> Optional[Path]:
    """Return the path to the latest 10-K folder for the given ticker under download_root."""
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
    """Find the main document (HTML/TXT) in the 10-K folder and return its full text."""
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


# ---------- Gemini 1.5 Flash: stable, generous free tier, good for large 10-K text ----------
GEMINI_MODEL = "gemini-2.0-flash"
# Wait 1 minute before retry when rate limited (free tier resets after a short period)
RATE_LIMIT_WAIT_SEC = 60
DELAY_BETWEEN_CALLS_SEC = 8


def get_gemini_model(api_key: str):
    """Return configured Gemini Flash model (generous free tier for large documents)."""
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
    """Call model.generate_content with retry on 429 (wait then retry up to max_retries times)."""
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


def get_ai_summary_and_report(
    api_key: str,
    full_text: str,
    item7_text: str,
    item8_text: str,
    ticker: str,
) -> tuple[str, str]:
    """
    Produce detailed analysis and CFA report. Only Item 7 and Item 8 are sent (pre-filtered).
    Sections are smart-chunked (head + tail) when long to keep token use low and avoid 429.
    """
    model = get_gemini_model(api_key)

    # Smart chunking: when over limit, keep head + tail (figures often at start/end)
    max_chars_per_section = 30000
    item7_text = smart_chunk(item7_text, max_chars=max_chars_per_section)
    item8_text = smart_chunk(item8_text, max_chars=max_chars_per_section)

    user_prompt = f"""You are a CFA charterholder and senior equity analyst. Use British English. Omit unnecessary qualifiers and filler; focus on figures, risks, and material facts.

Analyse the following 10-K excerpts for company ticker: {ticker}. The text below contains ONLY Item 7 (MD&A) and Item 8 (Financial Statements)—other sections have been pre-filtered out.

Use the provided text to produce a thorough, evidence-based analysis. Cite specific numbers and risk disclosures where relevant.

First, write a "DETAILED ANALYSIS" section with exactly three paragraphs (use subheadings):
1. **Financial Health**: Liquidity (current ratio, cash position, credit facilities), leverage (debt/equity, interest coverage), capital structure, and any covenant or refinancing risks. Cite figures from the statements.
2. **Profitability**: Revenue and earnings trends, margins (gross, operating, net), earnings quality (e.g. non-GAAP adjustments, one-time items), and sustainability of earnings. Use numbers from the 10-K.
3. **Key Risks**: Material risk factors from MD&A and notes (market, credit, operational, legal, ESG if material). Be specific; quote or paraphrase the filing.

Then, write a "CFA INVESTMENT REPORT" section in the style of a formal sell-side or buy-side investment memo. Include these subsections with clear headings:
- **Executive Summary**: 2–3 sentences on the company's position and your high-level view.
- **Investment Thesis**: Why an investor might consider this company (strengths, catalysts). Be specific.
- **Valuation Considerations**: What to watch (multiples, growth, margins, capital allocation). No exact price target required.
- **Key Risks to the Thesis**: Main downside risks that could invalidate the thesis.
- **Conclusion**: One short paragraph with a balanced wrap-up (e.g. Hold/Overweight/Underweight context and what would change your view).

Keep the entire response in British English. Use clear section headers (e.g. ## or **) and professional language."""

    full_content = f"""--- Item 7. Management's Discussion and Analysis (full or extended excerpt) ---

{item7_text}

--- Item 8. Financial Statements and Notes (full or extended excerpt) ---

{item8_text}

---

{user_prompt}"""

    try:
        response = _generate_with_retry(
            model,
            full_content,
            {"temperature": 0.3, "max_output_tokens": 8192},
        )
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise

    if not response or not response.text:
        return "No analysis generated.", "No report generated."

    text = response.text.strip()

    # Split into "DETAILED ANALYSIS" and "CFA INVESTMENT REPORT" if the model used those headers
    detailed = ""
    report = ""
    if "CFA INVESTMENT REPORT" in text.upper() or "CFA Investment Report" in text:
        parts = re.split(r"\n\s*(?:CFA INVESTMENT REPORT|CFA Investment Report)\s*\n", text, maxsplit=1, flags=re.IGNORECASE)
        detailed = (parts[0].replace("DETAILED ANALYSIS", "").strip() if parts else "").strip() or text
        report = parts[1].strip() if len(parts) > 1 else ""
        if not detailed:
            detailed = text
    else:
        detailed = text
        report = "(CFA Investment Report section not clearly separated; full analysis above.)"

    return detailed, report


def get_metrics_table_from_ai(api_key: str, item8_text: str, ticker: str) -> pd.DataFrame:
    """Extract Revenue, Net Income, Operating Cash Flow from Item 8 only (pre-filtered)."""
    model = get_gemini_model(api_key)
    excerpt = smart_chunk(item8_text, max_chars=25000)

    prompt = f"""You are a financial analyst. From the 10-K Item 8 excerpt below for company {ticker}, extract the following for the most recent 3–5 fiscal years. Focus only on figures; omit filler text.
- Revenue (or Net sales)
- Net Income (or Net earnings attributable to common shareholders)
- Cash flows from operating activities (Operating Cash Flow)

Reply with ONLY a single JSON object, no other text. Use fiscal years as keys (e.g. "2023", "2022", "2021").
Format:
{{"Revenue": {{"2023": 123.45, "2022": 100.0}}, "Net Income": {{"2023": 20.0, "2022": 18.0}}, "Operating Cash Flow": {{"2023": 25.0, "2022": 22.0}}}}
Use numbers in millions (e.g. 394328 for $394,328 million). If a value is not found, use null.

Item 8 excerpt:

{excerpt}"""

    try:
        response = _generate_with_retry(
            model,
            prompt,
            {"temperature": 0.1, "max_output_tokens": 1024},
        )
    except Exception as api_err:
        if _is_rate_limit_error(api_err):
            raise RuntimeError("Rate limit exceeded. Please try again in a few minutes.") from api_err
        raise

    if not response or not response.text:
        return pd.DataFrame()

    text = response.text.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return pd.DataFrame()
    try:
        data = json.loads(json_match.group())
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def download_and_extract_sections(ticker: str, email: str) -> tuple[str, str, str]:
    """Download 10-K, pre-filter, extract Item 7 & 8 only. Returns (full_text, item7, item8)."""
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
        item7 = smart_chunk(text_from_item7[:120000], max_chars=30000)
    if not item8:
        remainder = text_from_item7[100000:220000] if len(text_from_item7) > 100000 else text_from_item7
        item8 = smart_chunk(remainder, max_chars=30000)

    return full_text, item7, item8


def run_analysis(ticker: str, api_key: str, email: str, analysis_only: bool = False) -> tuple[str, str, str, pd.DataFrame]:
    """Download 10-K, extract Item 7/8, call Gemini; return summary, report, full_text, metrics table."""
    full_text, item7, item8 = download_and_extract_sections(ticker, email)

    detailed_summary, cfa_report = get_ai_summary_and_report(api_key, full_text, item7, item8, ticker)
    if analysis_only:
        df_metrics = pd.DataFrame()
    else:
        time.sleep(DELAY_BETWEEN_CALLS_SEC)
        df_metrics = get_metrics_table_from_ai(api_key, item8, ticker)

    return detailed_summary, cfa_report, full_text, df_metrics


# ---------- Streamlit UI ----------
st.set_page_config(page_title="10-K Financial Analyzer", layout="wide")
st.title("10-K Financial Analyzer")
st.caption("Download 10-K from SEC EDGAR; view detailed analysis and a CFA-style investment report. Powered by Google Gemini.")

with st.sidebar:
    st.header("Settings")
    google_api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Obtain from https://aistudio.google.com/apikey (Google AI Studio).",
    )
    email = st.text_input(
        "SEC EDGAR Email Address",
        value=os.environ.get("SEC_EDGAR_EMAIL", ""),
        help="Required for SEC programmatic download policy compliance.",
    )
    analysis_only = st.checkbox(
        "Analysis only (1 API call)",
        value=False,
        help="Skip metrics table to use only 1 API call. Turn on if you often hit rate limits.",
    )
    st.session_state["google_api_key"] = google_api_key
    st.session_state["email"] = email
    st.session_state["analysis_only"] = analysis_only

ticker = st.text_input("Stock Ticker (e.g. AAPL, MSFT)", value="AAPL", max_chars=10).strip().upper()
if not ticker:
    st.info("Enter a ticker and click 'Run Analysis', or pick one from the S&P 500 list below.")

# S&P 500 sample: (Company name, Ticker) – shown at bottom
SP500_SAMPLE = [
    ("Apple Inc.", "AAPL"), ("Microsoft Corporation", "MSFT"), ("Amazon.com Inc.", "AMZN"),
    ("NVIDIA Corporation", "NVDA"), ("Alphabet Inc. (Google)", "GOOGL"), ("Meta Platforms Inc. (Facebook)", "META"),
    ("Berkshire Hathaway Inc.", "BRK.B"), ("Tesla Inc.", "TSLA"), ("JPMorgan Chase & Co.", "JPM"),
    ("Visa Inc.", "V"), ("UnitedHealth Group Inc.", "UNH"), ("Procter & Gamble Co.", "PG"),
    ("Exxon Mobil Corporation", "XOM"), ("Johnson & Johnson", "JNJ"), ("Mastercard Inc.", "MA"),
    ("Chevron Corporation", "CVX"), ("Home Depot Inc.", "HD"), ("Merck & Co. Inc.", "MRK"),
    ("AbbVie Inc.", "ABBV"), ("Costco Wholesale Corporation", "COST"),
    ("PepsiCo Inc.", "PEP"), ("Coca-Cola Company", "KO"), ("Pfizer Inc.", "PFE"),
    ("Walmart Inc.", "WMT"), ("Netflix Inc.", "NFLX"), ("Adobe Inc.", "ADBE"),
    ("Salesforce Inc.", "CRM"), ("Comcast Corporation", "CMCSA"), ("Cisco Systems Inc.", "CSCO"),
    ("Oracle Corporation", "ORCL"), ("Intel Corporation", "INTC"), ("American Express Company", "AXP"),
    ("Bank of America Corp.", "BAC"), ("Wells Fargo & Company", "WFC"), ("Verizon Communications Inc.", "VZ"),
    ("AT&T Inc.", "T"), ("Disney (Walt Disney Co.)", "DIS"), ("Nike Inc.", "NKE"),
    ("McDonald's Corporation", "MCD"), ("Starbucks Corporation", "SBUX"), ("Goldman Sachs Group Inc.", "GS"),
    ("Morgan Stanley", "MS"), ("Boeing Company", "BA"), ("Caterpillar Inc.", "CAT"),
    ("3M Company", "MMM"), ("Honeywell International Inc.", "HON"), ("IBM (International Business Machines)", "IBM"),
    ("Qualcomm Inc.", "QCOM"), ("Texas Instruments Inc.", "TXN"), ("Amgen Inc.", "AMGN"),
    ("Gilead Sciences Inc.", "GILD"), ("Bristol-Myers Squibb Company", "BMY"), ("Eli Lilly and Company", "LLY"),
    ("Union Pacific Corporation", "UNP"), ("Lockheed Martin Corporation", "LMT"), ("Raytheon Technologies Corp.", "RTX"),
    ("Target Corporation", "TGT"), ("Lowe's Companies Inc.", "LOW"), ("Booking Holdings Inc.", "BKNG"),
    ("PayPal Holdings Inc.", "PYPL"), ("Broadcom Inc.", "AVGO"), ("Schlumberger Ltd.", "SLB"),
    ("ConocoPhillips", "COP"), ("Phillips 66", "PSX"),
    ("Ford Motor Company", "F"), ("General Motors Company", "GM"), ("General Electric Company", "GE"),
    ("FedEx Corporation", "FDX"), ("United Parcel Service Inc.", "UPS"), ("Delta Air Lines Inc.", "DAL"),
    ("American Airlines Group Inc.", "AAL"), ("Southwest Airlines Co.", "LUV"),
    ("Abbott Laboratories", "ABT"), ("Thermo Fisher Scientific Inc.", "TMO"), ("Danaher Corporation", "DHR"),
    ("Accenture plc", "ACN"), ("Intuit Inc.", "INTU"),
    ("ServiceNow Inc.", "NOW"), ("Workday Inc.", "WDAY"), ("Snowflake Inc.", "SNOW"),
    ("Zoom Video Communications Inc.", "ZM"), ("Spotify Technology S.A.", "SPOT"), ("Uber Technologies Inc.", "UBER"),
    ("Airbnb Inc.", "ABNB"), ("Moderna Inc.", "MRNA"), ("Regeneron Pharmaceuticals Inc.", "REGN"),
]

st.caption("Select a ticker above or choose from the list below.")

if st.button("Run Analysis"):
    if not ticker:
        st.error("Please enter or select a stock ticker.")
        st.stop()
    api_key = st.session_state.get("google_api_key", "")
    email = st.session_state.get("email", "")
    if not api_key:
        st.error("Please enter your Google API Key (Gemini) in Settings. You may also set GOOGLE_API_KEY in a .env file.")
        st.stop()
    if not email:
        st.error("Please enter your SEC EDGAR email address in Settings.")
        st.stop()

    analysis_only = st.session_state.get("analysis_only", False)
    try:
        with st.spinner("Step 1/2: Downloading 10-K and extracting Item 7 & 8 (selective sections only)..."):
            full_text, item7, item8 = download_and_extract_sections(ticker, email)

        with st.spinner("Step 2/2: Running Gemini analysis (typically 30–90s; if rate limited, we wait 60s then retry)..."):
            detailed_summary, cfa_report = get_ai_summary_and_report(api_key, full_text, item7, item8, ticker)
            if analysis_only:
                df_metrics = pd.DataFrame()
            else:
                time.sleep(DELAY_BETWEEN_CALLS_SEC)
                df_metrics = get_metrics_table_from_ai(api_key, item8, ticker)

        st.success("Analysis complete.")
        st.subheader("Detailed Analysis (Financial Health, Profitability, Key Risks)")
        st.markdown(detailed_summary)
        st.subheader("CFA Investment Report")
        st.markdown(cfa_report)
        st.subheader("Key Financial Metrics (Revenue, Net Income, Operating Cash Flow)")
        if not df_metrics.empty:
            st.dataframe(df_metrics, use_container_width=True)
        elif analysis_only:
            st.info("Metrics skipped (Analysis only mode). Turn off 'Analysis only' in Settings to fetch metrics.")
        else:
            st.info("No metrics extracted. Check the full Item 8 text.")
        with st.expander("View excerpt of extracted 10-K text"):
            st.text(full_text[:15000] + ("..." if len(full_text) > 15000 else ""))

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))
    except RuntimeError as e:
        st.error(str(e))
        if analysis_only:
            st.warning("You already have **Analysis only** on (1 API call). The limit is on Google's side — wait **2–5 minutes** without clicking, then press Run Analysis again.")
        else:
            st.info("Wait 2–5 minutes, then try again. Or enable 'Analysis only (1 API call)' in Settings to reduce usage.")
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or ("resource" in err_msg and "exhausted" in err_msg):
            st.error("Rate limit exceeded. Please try again in a few minutes.")
            if analysis_only:
                st.warning("You already have **Analysis only** on. Google's free tier limit is reached — wait **2–5 minutes**, then press Run Analysis again (no need to change settings).")
            else:
                st.info("Wait 2–5 minutes, then retry. Or enable **Analysis only (1 API call)** in the sidebar.")
        elif "404" in err_msg or "not found" in err_msg:
            st.error("The selected model is not available. Please try again later or check Google AI Studio for available models.")
        else:
            st.error("An error occurred. Please try again later.")
            st.caption("If the problem persists, check your API key and internet connection.")
        with st.expander("Error details (for troubleshooting)"):
            st.code(repr(e), language="text")
            st.caption("Share this with support if the issue continues.")

st.divider()
st.subheader("S&P 500 companies (sample) — Company name & Ticker")
st.caption("Click a row to copy the ticker, or type it in the box above.")
df_sp = pd.DataFrame(SP500_SAMPLE, columns=["Company name", "Ticker"])
with st.expander("Show list", expanded=True):
    st.dataframe(df_sp, use_container_width=True, hide_index=True)
