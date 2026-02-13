# 10-K Financial Analyzer

A web app that fetches the latest 10-K from SEC EDGAR for a given stock ticker, then uses **Item 7 (MD&A)** and **Item 8 (Financial Statements)** to produce a CFA-style analysis and key metrics. Powered by **Google Gemini**.

---

## Features

- **Selective section extraction:** Only Item 7 (MD&A) and Item 8 (Financial Statements) are sent to the API; PART I and Items 1–6 are pre-filtered to reduce tokens.
- **Smart chunking:** Long sections are trimmed to head + tail to stay within token limits while keeping high-signal content.
- **Two-step progress:** The UI shows Step 1 (download + extract) and Step 2 (Gemini analysis) so you can see where time is spent.
- **Analysis only mode:** Optional single API call (summary + CFA report only) to reduce rate-limit issues.
- **S&P 500 reference list:** A table of company names and tickers (sample) is shown at the bottom of the page for quick lookup.

**Typical run time:** About **1–2 minutes** (roughly 1 minute with “Analysis only” enabled; up to 2 minutes with metrics). If the API is rate-limited, the app waits 60 seconds and retries automatically.

---

## Tech Stack

- **UI**: Streamlit  
- **Data**: sec-edgar-downloader (SEC EDGAR)  
- **AI**: Google Gemini (google-generativeai)  

---

## Technical Challenge: Handling Large-Scale Financial Filings

During the initial development, I encountered a **429 Resource Exhausted** error due to the massive size of 10-K filings exceeding the LLM's token quota and rate limits.

**Consultation & Architectural Pivot:**  
After consulting with a senior software engineer, I re-architected the application to optimize token usage. Instead of processing the entire document, I implemented a **"Selective Section Extraction"** strategy.

**Implemented Solution:**

- **Targeted Parsing:** Developed a regex-based parser to isolate only critical sections: Item 7 (MD&A) and Item 8 (Financial Statements).
- **Token Optimization:** Integrated a "Chunking & Filtering" logic to remove boilerplate legal text, sending only high-signal data to the Gemini API.
- **Efficiency:** This reduced token consumption by **over 80%**, ensuring stable performance within free-tier limits while maintaining analytical depth.

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

- **Selective extraction & pre-filtering:** Regex-based extraction of Item 7 and Item 8 only; content before Item 7 is dropped to cut token use.
- **Smart chunking:** Sections over ~30k characters are reduced to head + tail before sending to Gemini.
- **Progress steps:** Step 1 (download + extract) and Step 2 (Gemini analysis, ~30–90s) with clear spinner messages.
- **S&P 500 list:** Bottom of the page now includes a sample table of S&P 500 companies with **company name** and **ticker** for easy reference.
- **Run time:** Results usually appear within 1–2 minutes under normal conditions.

---

## License and Disclaimer

This project is for learning and portfolio use. Comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support) when using SEC data and with Google's terms for the Gemini API.
