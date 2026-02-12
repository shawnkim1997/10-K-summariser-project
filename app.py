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


def find_item_section(text: str, item_num: int, title_keywords: list) -> str:
    """Find Item N section (e.g. item_num=7 -> Item 7, item_num=8 -> Item 8)."""
    pattern = re.compile(
        r"\bItem\s+" + str(item_num) + r"\b[.\s]*[^\n]*(" + "|".join(re.escape(k) for k in title_keywords) + r")?",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    next_item = re.search(r"\n\s*Item\s+\d+\s+", text[start + 50 :], re.IGNORECASE)
    if next_item:
        end = start + 50 + next_item.start()
    else:
        end = min(start + 150000, len(text))  # Cap section size to stay within token limits
    return text[start:end].strip()


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
    Use Gemini 1.5 Flash to produce:
    1) A detailed three-part summary (financial health, profitability, key risks).
    2) A CFA Investment Report-style section (Executive Summary, Investment Thesis, Risks, etc.).
    Only Item 7 and Item 8 are sent; sections are trimmed to avoid token/rate limits.
    """
    model = get_gemini_model(api_key)

    # Keep payload smaller to reduce token usage and avoid 429 rate limits
    max_chars_per_section = 40000
    if len(item7_text) > max_chars_per_section:
        item7_text = item7_text[:max_chars_per_section] + "\n\n[ ... section truncated ... ]"
    if len(item8_text) > max_chars_per_section:
        item8_text = item8_text[:max_chars_per_section] + "\n\n[ ... section truncated ... ]"

    user_prompt = f"""You are a CFA charterholder and senior equity analyst writing for an accounting and finance audience. Your analysis must be evidence-based, cite specific figures from the 10-K where relevant, and follow professional investment report standards. Use British English throughout (e.g. analyse, summarise, colour, favour, organisation).

Analyse the following 10-K content for company ticker: {ticker}.

Use the FULL text provided below (Item 7 MD&A and Item 8 Financial Statements) to produce a thorough, detailed analysis. Do not summarise superficially—reference specific numbers, trends, and risk disclosures.

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
    """Ask Gemini to extract Revenue, Net Income, Operating Cash Flow by year from Item 8 and return a table."""
    model = get_gemini_model(api_key)
    max_item8_chars = 25000
    excerpt = item8_text[:max_item8_chars] if len(item8_text) > max_item8_chars else item8_text

    prompt = f"""You are a financial analyst. From the 10-K Item 8 excerpt below for company {ticker}, extract the following for the most recent 3–5 fiscal years (if available):
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


def run_analysis(ticker: str, api_key: str, email: str, analysis_only: bool = False) -> tuple[str, str, str, pd.DataFrame]:
    """Download 10-K, extract text, get Item 7/8, return detailed summary, CFA report, and optionally metrics table."""
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

    item7 = find_item_section(full_text, 7, ["Management's Discussion", "MD&A", "Analysis"])
    item8 = find_item_section(full_text, 8, ["Financial Statements", "Consolidated"])

    # Use only extracted Item 7 / Item 8; fallback to trimmed full text if sections not found
    if not item7:
        item7 = full_text[:80000]
    if not item8:
        item8 = full_text[80000:160000] if len(full_text) > 80000 else full_text[:80000]

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
    st.info("Enter a ticker and click 'Run Analysis'.")
    st.stop()

if st.button("Run Analysis"):
    api_key = st.session_state.get("google_api_key", "")
    email = st.session_state.get("email", "")
    if not api_key:
        st.error("Please enter your Google API Key (Gemini) in Settings. You may also set GOOGLE_API_KEY in a .env file.")
        st.stop()
    if not email:
        st.error("Please enter your SEC EDGAR email address in Settings.")
        st.stop()

    analysis_only = st.session_state.get("analysis_only", False)
    with st.spinner("Downloading 10-K and running Gemini analysis (if rate limited, waiting up to 60s before retry)..."):
        try:
            detailed_summary, cfa_report, full_text, df_metrics = run_analysis(ticker, api_key, email, analysis_only=analysis_only)
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
