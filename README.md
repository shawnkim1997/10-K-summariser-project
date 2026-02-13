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
├── app.py              # Streamlit app (Gemini)
├── requirements.txt    # Python dependencies
├── .env.example        # Example env vars (copy to .env)
├── README.md           # This file
└── TECHNICAL_NOTES.md  # Technical challenge & solution (for reference)
```

---

## Recent updates

- **Hybrid architecture:** Item 7 (MD&A) only is sent to Gemini for qualitative analysis (strategy, risks, sentiment). Financial metrics (Revenue, Net Income, Operating Cash Flow) come from **yfinance**—no Item 8 to the AI, fewer tokens, and accurate numbers.
- **HTML cleansing:** Pre-LLM step strips HTML remnants, extra whitespace, and page numbers to compress Item 7 text before sending to Gemini.
- **Selective extraction:** Regex-based extraction of Item 7 only for the API; content before Item 7 is dropped.
- **Smart chunking:** Item 7 over ~20k characters is reduced to head + tail before sending to Gemini.
- **Progress steps:** Step 1 (download + extract Item 7), Step 2 (Gemini qualitative analysis + yfinance metrics) with clear spinner messages.
- **S&P 500 list:** Sample table of S&P 500 companies (company name and ticker) at the bottom for quick reference.
- **Run time:** One Gemini call plus instant yfinance data; results typically within 1–2 minutes under normal conditions.

---

## License and Disclaimer

This project is for learning and portfolio use. Comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support) when using SEC data and with Google's terms for the Gemini API.
