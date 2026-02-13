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

업데이트 내용을 **날짜 순(최신 → 과거)**으로 정리했습니다.

| 날짜 (Date) | 업데이트 내용 (Updates) |
|-------------|-------------------------|
| **2025-02-12** | **하이브리드 아키텍처:** Item 7(MD&A)만 Gemini로 전송, 재무 지표는 **yfinance**로 조회. **HTML 클렌징** 함수 추가(태그·공백·페이지 번호 제거). 프롬프트를 경영 전략·리스크·감성(Sentiment) 분석 중심으로 변경. **find_toc.py** 추가(SEC EDGAR HTML URL → 목차 추출 스크립트). README를 하이브리드 구조·Tech Stack·Technical Challenge 기준으로 전면 수정. |
| **2025-02-12** | README에 **Update history (Changelog)** 섹션 추가 — 날짜별 업데이트 순서 정리. Project Structure에 `find_toc.py` 반영. |
| **2025-02-01** (경과) | Selective extraction: Item 7·8만 추출해 API 전송. 429 대응: 재시도(60초 대기), "Analysis only" 옵션, 타임아웃·404·에러 메시지 정리. Google **Gemini** 전환 및 `GOOGLE_API_KEY`(.env·사이드바) 지원. CFA Investment Report·지표 테이블·스피너 2단계 표시. |
| **2025-01-XX** (경과) | 초기 버전: SEC EDGAR 10-K 다운로드, Item 7·8 추출, LLM 분석, Streamlit UI. S&P 500 샘플 목록(회사명·티커) 추가. |

---

## License and Disclaimer

This project is for learning and portfolio use. Comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support) when using SEC data and with Google's terms for the Gemini API.
