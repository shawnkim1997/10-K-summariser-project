# All-in-One Financial Analysis Dashboard

A **cost-effective** Streamlit app that unifies **qualitative AI-driven insights** and **quantitative valuation** in a single workflow. **Hybrid architecture:** Gemini powers narrative analysis (10-K MD&A and Risk Factors); all numbers—DCF inputs and peer multiples—come from **yfinance**, keeping API costs low and numerical accuracy high.

The app is organised into **three tabs:**

| Tab | Purpose |
|-----|---------|
| **1. 10-K Qualitative Insights** | SEC EDGAR 10-K → Item 1A (Risk Factors) + Item 7 (MD&A) → cleaned text → Gemini. Output: management’s tone (sentiment), key strategic shifts, and major hidden risks. |
| **2. DCF Valuation** | yfinance for FCF, Debt, Cash, Shares. Sliders for WACC, terminal growth, FCF growth. **3-scenario model** (Bull / Base / Bear) with intrinsic value per share. No LLM. |
| **3. Industry Comps** | Comma-separated competitor tickers → yfinance **Forward P/E**, **EV/EBITDA**, **P/B** → comparison table. |

---

## Features

- **Tab 1 — 10-K Qualitative Insights:** Item 1A + Item 7 from SEC EDGAR; HTML cleaned; one Gemini call for tone, strategic shifts, and hidden risks.
- **Tab 2 — DCF Valuation:** FCF, Debt, Cash, Shares from yfinance; WACC / terminal growth / FCF growth sliders; Bull/Base/Bear intrinsic value per share; no LLM.
- **Tab 3 — Industry Comps:** Comma-separated tickers; Forward P/E, EV/EBITDA, P/B from yfinance; comparison table.
- **Error handling:** Try/except for SEC EDGAR and yfinance; clear messages when data is missing or requests fail.
- **UI:** Three-tab layout, sidebar for API key and SEC email, professional layout.

**Run time:** Tab 1 ≈ 1–2 min (one Gemini call); Tabs 2–3 use yfinance (seconds). Rate limit: 60s retry.

---

## Project Origin & Vision

### The Origin — The Walk

The core idea for this all-in-one architecture came during a **quiet walk**. I was deep in thought about the inefficiencies and fragmentation of traditional equity research: narrative buried in 200-page filings, valuation models in separate spreadsheets, and comps scattered across different tools. It became clear that what we need is not more dashboards, but **one seamless workflow**—where qualitative AI insights and quantitative valuation models live in the same place, speak the same language, and serve the same decision. That moment crystallised into the design you see here: **unified, cost-conscious, and built for the analyst who thinks in both words and numbers.**

### The Vision — Commercialization

This repository is a **functional MVP (Minimum Viable Product)** and **demo**. It proves the concept: hybrid architecture works; 10-K + DCF + comps can sit in a single interface; and the unit economics (one Gemini call for narrative, free data for the rest) scale. The code is production-minded but not yet productised—it is the foundation on which a commercial product will be built.

### Future Roadmap

The **ultimate goal** is to launch this as a **fully commercialised B2C/B2B SaaS** application. We aim to serve **retail investors** who want institutional-grade structure without the complexity, and **finance professionals** (equity analysts, portfolio managers, corporate development) who want to move from filing → insight → valuation in one flow. Data-driven, transparent, and built by someone who cares as much about the quality of the analysis as the quality of the code. This project is the first step on that path.

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

**1. Go to the project folder**
```bash
cd "/Users/seonpil/Documents/FQDC Project"
```

**2. Activate the virtual environment** (required so `pip` and `streamlit` are found)
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
After activation, your prompt usually shows `(venv)`.

**3. Install dependencies** (only needed once, or when requirements change)
```bash
pip install -r requirements.txt
```

**4. Start the app**
```bash
streamlit run app.py
```

If you don’t have a `venv` folder yet, create it first:
```bash
python3 -m venv venv
source venv/bin/activate   # then steps 3 and 4
```

Open the sidebar to set **Google API Key** and **SEC EDGAR Email**, then use the three tabs (10-K Insights, DCF, Comps) as needed.

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
