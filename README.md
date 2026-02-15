# All-in-One Financial Analysis Dashboard

A **cost-effective** Streamlit app that unifies **qualitative AI-driven insights** and **quantitative valuation** in a single workflow. **Hybrid architecture:** Gemini powers narrative analysis (10-K MD&A and Risk Factors); all numbers—DCF inputs and peer multiples—come from **yfinance**, keeping API costs low and numerical accuracy high.

The app is organised into **eight tabs:**

| Tab | Purpose |
|-----|---------|
| **1. 10-K & MD&A Insights** | SEC EDGAR 10-K → Item 1A + Item 7 → cleaned text → Gemini. DuPont, Altman Z, red flags, YoY; Sankey, Radar, 5Y financials, KPI. |
| **2. Market Heatmap** | Sector and macro heatmaps; rates, oil, VIX and related indicators. |
| **3. Valuation Hub (DCF / RIM)** | 10-year 2-stage DCF, Reverse DCF, RIM. Bull/Base/Bear; ticker-currency display and USD conversion when non-USD. |
| **4. Industry Analysis & Comps** | Peer comps (P/E, EV/EBITDA, P/B); conditional formatting; **Generate Industry Outlook** (Gemini) for macro trends. |
| **5. SEC Filings (Raw)** | SEC filing list and links to full documents. |
| **6. Earnings & Estimates** | Consensus, Beat/Miss chart, analyst targets and recommendations; ticker-currency and USD conversion. |
| **7. Portfolio & Watchlist** | Holdings with **per-position currency** (USD/GBP/EUR/KRW/JPY/CNY), **fractional quantity**, AI screenshot import (Gemini Vision), **FX-adjusted returns**. |
| **8. Crypto** | Bithumb (KRW) and Binance (USD) live prices. |

---

## Features

- **Tab 1 — 10-K & MD&A:** Item 1A + Item 7; DuPont, Altman Z, red flags, YoY; sector/industry badge; sector-specific metrics (Tech/Retail/Financials); Gemini comparative MD&A with sector-aware Non-GAAP KPI table. TTM fallback and N/A handling when yfinance rows are missing.
- **Tab 2 — Market Heatmap:** Sector and macro heatmaps; key indicators.
- **Tab 3 — Valuation Hub:** 10-year 2-stage DCF, Reverse DCF, RIM. Smart defaults (Beta/CAPM WACC, terminal growth). Reference panel: analyst consensus and Damodaran sector WACC/ERP/Rf. **Ticker-currency display** and **USD conversion** for non-USD names (`format_price_with_usd`).
- **Tab 4 — Sector Analysis:** Predefined sectors with peer tickers; comps table; green/red formatting; AI Industry Outlook (Gemini).
- **Tab 5 — SEC Filings:** Raw filing list and links.
- **Tab 6 — Earnings & Estimates:** Consensus, Beat/Miss chart, analyst targets; **ticker currency** and USD conversion.
- **Tab 7 — Portfolio & Watchlist:** **Per-position Currency** (USD, GBP, EUR, KRW, JPY, CNY). **Fractional quantity** (e.g. 30.395107). **AI screenshot import** (Gemini Vision): extracts Ticker, Average Price, Currency (from $, £, €, ₩, ¥), Quantity (decimals preserved). **FX-adjusted returns:** user cost in local currency → real-time FX → `adjusted_avg` in asset currency → **Total Return %** = (current − adjusted_avg) / adjusted_avg. `get_fx_rate`, `get_fx_rate_to_usd` (yfinance FX, TTL 60–120s).
- **Tab 8 — Crypto:** Bithumb KRW, Binance USD.
- **App-wide multi-currency:** `get_currency_for_ticker` (auto-detect trading currency); `format_price_with_usd` (local + USD equivalent, e.g. ₩ 181,200 (≈ $ 132.50)).
- **Error handling:** Try/except for SEC EDGAR, yfinance, and Gemini; clear messages and optional manual overrides.
- **UI:** Sidebar (API key, SEC email, **global company search** via yahooquery—search by name in any language). **Quantitative charts** (Sankey, Radar, F-Score) can use SEC 10-K Item 8 + Gemini extraction (US) or yahooquery/yfinance (global tickers with auto suffix).

**Run time:** Tab 1 ≈ 1–2 min (one Gemini call); other tabs use yfinance (seconds). Rate limit: 60s retry.

---

## Project Origin & Vision

### The Origin — The Walk

The core idea for this all-in-one architecture came during a **quiet walk**. I was deep in thought about the inefficiencies and fragmentation of traditional equity research: narrative buried in 200-page filings, valuation models in separate spreadsheets, and comps scattered across different tools. It became clear that what we need is not more dashboards, but **one seamless workflow**—where qualitative AI insights and quantitative valuation models live in the same place, speak the same language, and serve the same decision. That moment crystallised into the design you see here: **unified, cost-conscious, and built for the analyst who thinks in both words and numbers.**

### The Vision — Commercialization

This repository is a **functional MVP (Minimum Viable Product)** and **demo**. It proves the concept: hybrid architecture works; 10-K + DCF + comps can sit in a single interface; and the unit economics (one Gemini call for narrative, free data for the rest) scale. The code is production-minded but not yet productised—it is the foundation on which a commercial product will be built.

### Future Roadmap

The **ultimate goal** is to launch this as a **fully commercialised B2C/B2B SaaS** application. We aim to serve **retail investors** who want institutional-grade structure without the complexity, and **finance professionals** (equity analysts, portfolio managers, corporate development) who want to move from filing → insight → valuation in one flow. Data-driven, transparent, and built by someone who cares as much about the quality of the analysis as the quality of the code. This project is the first step on that path.

---

## Design Rationale & Interview Notes

*(Why certain features were built the way they were — useful for interviews and discussions.)*

- **Undergraduate automation mindset**  
  As an undergraduate student, I realised that rather than just learning Excel and basic Python and doing everything manually, **automating the full workflow with AI and programmatic data** is far more powerful. This dashboard is the result: one place for 10-K narrative (Gemini), numbers (yfinance), DCF, and comps, so the analyst can focus on judgment instead of copy-pasting between tools.

- **Why a 10-year DCF instead of 5 years**  
  A standard 5-year projection is often **too short for practical, real-world corporate analysis**. Many companies have growth that extends beyond five years, and terminal value then dominates the result, which can overstate or misstate value. The **10-year 2-stage model** (Stage 1: Years 1–5 at the chosen FCF growth rate; Stage 2: Years 6–10 with growth **linearly fading** down to the terminal growth rate) is closer to how institutional DCFs are built and avoids absurd valuations for high-growth names.

- **Integrating Damodaran's academic baselines**  
  I regularly read valuation literature and **wanted to integrate Aswath Damodaran's academic baselines directly into the app**. The "Reference: Analyst & Macro Assumptions" panel shows sector WACC benchmarks (e.g. Software 8.5%, Retail 7.5%, Hardware 9.0%, Financials 8.0%), US equity risk premium (~4.6%), and the 10-year risk-free rate (~4.2%), with a link to his data and methodology so users can verify and align their assumptions with established research.

- **Consensus numbers next to the DCF sliders**  
  Having **analyst consensus data (target price, recommendation, revenue/earnings growth) right next to the DCF sliders** makes it much easier to make informed adjustments. Instead of guessing WACC or growth, the user can compare their inputs to both consensus and Damodaran's macro baselines in one view, like a professional equity research dashboard.

- **Commercialization**  
  Once the app's **completeness and robustness reach a higher professional standard**, my ultimate goal is to **fully commercialise it** (e.g. B2C/B2B SaaS). The current codebase is built as a production-minded MVP and demo to validate the hybrid architecture and user flow before scaling.

---

## Tech Stack

- **UI**: Streamlit  
- **Data**: sec-edgar-downloader (SEC EDGAR), **yahooquery** (search + fundamentals), **yfinance** (prices, FX, fallback)  
- **AI**: Google Gemini (google-generativeai)  
- **Parsing / cleansing**: BeautifulSoup, lxml, regex  

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
cd "/path/to/your/FQDC Project"
```
*(Replace with your actual project path.)*

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

If you don't have a `venv` folder yet, create it first:
```bash
python3 -m venv venv
source venv/bin/activate   # then steps 3 and 4
```

Open the sidebar to set **Google API Key** and **SEC EDGAR Email**, then use the eight tabs (10-K Insights, Heatmap, DCF, Comps, SEC Filings, Earnings, Portfolio, Crypto) as needed.

---

## Project Structure

```
├── app.py              # Streamlit app (Gemini, hybrid flow)
├── find_toc.py         # Standalone script: find Table of Contents from SEC EDGAR HTML URL
├── requirements.txt    # Python dependencies
├── .env.example        # Example env vars (copy to .env)
├── README.md           # This file
└── TECHNICAL_NOTES.md  # Technical challenge & solution (for reference)
```

---

## Update history (Changelog)

Updates are listed in **reverse chronological order (newest first)**. Each row summarises **what** was added and **why** (where relevant).

| Date (UTC) | Updates |
|------------|---------|
| **2025-02-15** | **Multi-currency portfolio & app-wide FX:** (1) **Portfolio (Tab 7):** Per-position **Currency** column (USD, GBP, EUR, KRW, JPY, CNY). **Fractional quantity** support (e.g. 30.395107). **FX-adjusted returns:** user cost/currency → real-time `get_fx_rate(user_curr, stock_curr)` → `adjusted_avg` in asset currency → Total Return % = (current − adjusted_avg) / adjusted_avg. (2) **AI screenshot (Gemini Vision):** Prompt updated to extract **Currency** (from $, £, €, ₩, ¥ → USD/GBP/EUR/KRW/JPY) and **Quantity** with decimals preserved. (3) **App-wide:** `get_currency_for_ticker`, `get_fx_rate`, `get_fx_rate_to_usd` (yfinance FX, TTL 60–120s), `format_price_with_usd` (local + USD e.g. ₩ 181,200 (≈ $ 132.50)). (4) **Valuation Hub:** DCF/Reverse DCF/RIM show ticker currency and USD conversion when non-USD. (5) **Earnings & Estimates:** Analyst targets/price in ticker currency with USD conversion. README restored to English with full changelog and 8-tab layout. |
| **2025-02-14** | **Global company search & README:** Sidebar company search replaced with yahooquery `search()`: type company name (e.g. Samsung, 삼성, Mitsubishi), click "Search Company", select from dropdown `[Exchange] Symbol - Name`. Filter: EQUITY/ETF only (exclude INDEX/MUTUALFUND). Market suffix inferred from symbol (.KS/.KQ, .T, .L). README: Tech Stack (yahooquery, lxml), Features (global search, Item 8 quant), run/push instructions path-agnostic. |
| **2025-02-13** | **Design Rationale & README:** New section "Design Rationale & Interview Notes" (undergrad automation mindset, 10y DCF rationale, Damodaran integration, consensus-panel rationale, commercialization). README Features and tab table updated to reflect 10Y 2-stage DCF, sector analysis, and Wall Street Assumptions panel. Changelog expanded with more detailed entries. |
| **2025-02-13** | **Institutional DCF & Wall Street panel:** (1) **10-year 2-stage DCF:** Stage 1 (Y1–5) at user FCF growth; Stage 2 (Y6–10) linear fade from that rate to terminal growth (avoids absurd valuations for high-growth stocks). TV at Year 10; all FCFs + TV discounted to PV. (2) **Wall Street Assumptions panel** (expander below sliders): **Left column** — Analyst consensus from yfinance: target mean price, recommendation, revenue growth est., earnings growth est. (N/A if missing). **Right column** — Damodaran macro baseline: sector WACC map (Software 8.5%, Retail 7.5%, Hardware 9.0%, Financials 8.0%, etc.), US ERP ~4.6%, 10Y risk-free ~4.2%, plus markdown link to his WACC data page for methodology. Company sector matched via `get_sector_industry` for Damodaran WACC. |
| **2025-02-13** | **Smart DCF defaults:** Slider defaults no longer hardcoded. **WACC:** CAPM approximation using `ticker.info.get('beta')` (default 1.0), Risk-free 4%, MRP 5%; default WACC = 4 + Beta×5, rounded to 1 decimal. **Terminal growth:** Fixed at 2.5% (Damodaran-style, long-term US GDP). **FCF growth:** From `revenueGrowth` or `earningsGrowth` (e.g. 0.15 → 15%); fallback 8%. Caption above sliders: "Slider defaults are auto-generated based on the company's Beta (CAPM) and revenue growth estimates." |
| **2025-02-13** | **Robust DCF data & comps:** (1) **Shares/Debt/Cash:** Multi-step fallback (fast_info → info → balance sheet) so S&P 500 names rarely need manual input. Shares: `fast_info.shares` → `sharesOutstanding` → `impliedSharesOutstanding`; display as "X.XXB Shares (real-time, auto-fetched)". Manual number_input only when all sources fail. (2) **Tab 3 redesign — Top-down sector analysis:** Manual ticker input removed. `SECTORS` dict (e.g. Semiconductors, Software & Cloud, Consumer Retail, Financials, Healthcare) with top 5 tickers each; st.selectbox to choose industry; comps table auto-loads with spinner. yfinance keys fixed to `forwardPE`, `enterpriseToEbitda`, `priceToBook`; missing shown as N/A. Conditional formatting: lowest P/E and EV/EBITDA green, highest red. **Generate Industry Outlook** button: Gemini prompt for macro analyst-style report (12–18 month trends, growth drivers, headwinds/regulatory risks); report rendered in Markdown below table. |
| **2025-02-13** | **Bulletproof DCF & Excel-style logic:** DCF no longer fails when yfinance misses data. Base FCF = OCF − CapEx; if Shares/Debt/Cash missing, st.number_input fallbacks. Three sliders (WACC, Terminal Growth, Projected FCF Growth) drive full DCF; intrinsic value vs current price (from yfinance) and Bull/Base/Bear table. Tab 1: Interest Coverage "nan%" fixed (N/A when Interest Expense 0 or missing). |
| **2025-02-12 15:30** | **Dynamic Sector-Specific Analysis:** DuPont table None/NaN → "N/A". Sector & industry badge (Tab 1). Sector-specific metrics: Tech (Rule of 40, FCF margin, R&D % revenue), Retail (inventory turnover, operating margin), Financials (ROE, ROA). Tab 2 caption for Financials (FCF/EBITDA less relevant). Gemini MD&A: sector/industry passed in; prompt asks for industry-specific Non-GAAP KPIs in a markdown table. |
| **2025-02-12 11:00** | **Data robustness & TTM fallback:** `_get_row_series` try/except; `_na(x)` for display. TTM fallback when annual financials/balance_sheet missing (quarterly sum / latest quarter). `get_sector_industry(ticker)` added. DuPont/Altman return empty dict on exception. |
| **2025-02-12 09:00** | **Hybrid architecture:** Item 7 only to Gemini; yfinance for numbers. HTML cleansing, find_toc.py. Prompt: strategy, risks, sentiment. |
| **2025-02-12 08:45** | Changelog and find_toc.py in Project Structure. |
| **2025-02-12** | **Remember API key & email:** Optional "Remember API key & email (save locally)" checkbox; values stored in `.app_prefs.json` (in .gitignore); prefill on load; uncheck removes file. |
| **2025-02-12** | **S&P 500 sample expander removed** from sidebar (user request). |
| **2025-02-01** (approx.) | Item 7 & 8 selective extraction; 429 retry 60s; Gemini, GOOGLE_API_KEY; CFA report, metrics table. |
| **2025-01-XX** (approx.) | Initial release: SEC EDGAR 10-K, Item 7 & 8, LLM analysis, Streamlit UI, S&P 500 sample list. |

---

## Push to GitHub

From the project folder, commit and push (run in your terminal so authentication works):

```bash
cd "/path/to/your/FQDC Project"
git add README.md app.py requirements.txt
git status
git commit -m "README: restore English, full changelog; add 2025-02-15 multi-currency portfolio & FX"
git push origin main
```

If you use another branch or remote: replace `main` or `origin`. New repo: `git init`, then `git remote add origin <your-repo-url>`.

---

## License and Disclaimer

This project is for learning and portfolio use. Comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support) when using SEC data and with Google's terms for the Gemini API.
