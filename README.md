# 10-K Financial Analyzer

A web app that fetches the latest 10-K from SEC EDGAR for a given stock ticker and uses a **hybrid architecture**: **qualitative** analysis (Item 7 MD&A only) via **Google Gemini**, and **quantitative** metrics (Revenue, Net Income, Operating Cash Flow) from **yfinance**. CFA-style report and key financials in one place.

---

## Features

- **Hybrid processing (qualitative + quantitative):**
  - **Qualitative:** Only **Item 7 (MD&A)** is sent to Gemini for analysis of management’s strategy, market risks, and sentiment—no Item 8 (financial statements) to the AI, which cuts token use and avoids number hallucination.
  - **Quantitative:** Financial metrics (Revenue, Net Income, Operating Cash Flow) are fetched directly from **yfinance**—fast, accurate, and no extra API tokens.
- **HTML cleansing:** Before sending Item 7 to the LLM, the app strips remaining HTML tags, collapses whitespace, and removes page numbers to compress tokens.
- **Selective extraction:** The 10-K is parsed with regex; only content from Item 7 onward is used for AI; PART I and Items 1–6 are dropped.
- **Smart chunking:** Long Item 7 text is trimmed to head + tail to stay within token limits.
- **Two-step progress:** Step 1 (download + extract Item 7), Step 2 (Gemini analysis + yfinance metrics).
- **Analysis only mode:** Optional hide for the metrics table (Gemini still runs once on Item 7).
- **S&P 500 reference list:** Sample table of company names and tickers at the bottom for quick lookup.

**Typical run time:** About **1–2 minutes** (one Gemini call; yfinance metrics are near-instant). If the API is rate-limited, the app waits 60 seconds and retries automatically.

---

## Tech Stack

- **UI**: Streamlit  
- **Data**: sec-edgar-downloader (SEC EDGAR), **yfinance** (financial metrics)  
- **AI**: Google Gemini (google-generativeai)  
- **Parsing / cleansing**: BeautifulSoup, regex  

---

## Technical Challenge: Handling Large-Scale Financial Filings

During the initial development, I encountered a **429 Resource Exhausted** error due to the massive size of 10-K filings exceeding the LLM's token quota and rate limits.

**Consultation & Architectural Pivot:**  
After consulting with a senior software engineer, I re-architected the application to optimize token usage. The current design uses a **hybrid architecture** that separates qualitative and quantitative work.

**Implemented Solution:**

- **Selective section extraction:** A regex-based parser isolates Item 7 (MD&A) only for the AI; Item 8 is no longer sent to the LLM.
- **Hybrid processing:**  
  - **Qualitative (Gemini):** Item 7 only—strategy, risks, and sentiment. This drastically reduces tokens and avoids AI errors on exact figures.  
  - **Quantitative (yfinance):** Revenue, Net Income, and Operating Cash Flow are pulled from yfinance, so numbers are accurate and no tokens are spent on financial tables.
- **HTML cleansing:** Before sending Item 7 to Gemini, the app runs a cleansing step (BeautifulSoup + regex) to strip tags, collapse whitespace, and drop page numbers, further compressing tokens.
- **Chunking:** Long Item 7 text is trimmed to head + tail to stay within token limits.
- **Efficiency:** Token consumption is greatly reduced (one API call; no Item 8 in the prompt), and numeric accuracy is guaranteed via yfinance.

For full technical notes and code references, see **[TECHNICAL_NOTES.md](./TECHNICAL_NOTES.md)**.

---

## Requirements

- Python 3.9+  
- [Google API Key (Gemini)](https://aistudio.google.com/apikey)  
- An email address for SEC EDGAR (required for programmatic access)  
- Optional: `.env` with `GOOGLE_API_KEY` and `SEC_EDGAR_EMAIL`  

---

## How to Run

```bash
cd "/path/to/FQDC Project"
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the sidebar to set **Google API Key** and **SEC EDGAR Email**, then enter a ticker (e.g. `AAPL`, `MSFT`) and click **Run Analysis**. Use **Analysis only (1 API call)** if you hit rate limits.

---

## Project Structure

```
├── app.py              # Streamlit app (Gemini, hybrid flow)
├── find_toc.py         # Standalone script: find Table of Contents from SEC EDGAR HTML URL
├── requirements.txt    # Python dependencies
├── .env.example        # Example env vars (copy to .env)
├── README.md           # This file
└── TECHNICAL_NOTES.md # Technical challenge & solution (for reference)
```

---

## Update history (Changelog)

Updates are listed in **reverse chronological order (newest first)**.

| Date | Updates |
|------|---------|
| **2025-02-12** | **Hybrid architecture:** Send only Item 7 (MD&A) to Gemini; fetch financial metrics from **yfinance**. Add **HTML cleansing** (strip tags, collapse whitespace, remove page numbers). Change prompt to focus on management strategy, risks, and **sentiment** analysis. Add **find_toc.py** (SEC EDGAR HTML URL → Table of Contents extraction). Overhaul README for hybrid flow, Tech Stack, and Technical Challenge. |
| **2025-02-12** | Add **Update history (Changelog)** to README and include `find_toc.py` in Project Structure. |
| **2025-02-01** (approx.) | Selective extraction: send only Item 7 & 8 to the API. 429 handling: retry with 60s wait, "Analysis only" option, timeout/404/error messages. Switch to Google **Gemini** and add `GOOGLE_API_KEY` (.env and sidebar). CFA Investment Report, metrics table, two-step spinners. |
| **2025-01-XX** (approx.) | Initial release: SEC EDGAR 10-K download, Item 7 & 8 extraction, LLM analysis, Streamlit UI. S&P 500 sample list (company name and ticker). |

---

## License and Disclaimer

This project is for learning and portfolio use. Comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support) when using SEC data and with Google's terms for the Gemini API.
