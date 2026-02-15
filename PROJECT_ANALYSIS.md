# FQDC 프로젝트 심층 분석 보고서

> All-in-One Financial Analysis Dashboard — 정성·정량 하이브리드 금융 분석 대시보드

---

## 1. 프로젝트 정체성 및 비전

### 1.1 목적
- **문제:** 주식 리서치의 분산 — 200페이지급 10-K 공시, 별도 스프레드시트 DCF, 여러 도구에 흩어진 동종사 비교.
- **해결:** **단일 워크플로우**에서 공시 → 인사이트 → 밸류에이션 → 동종사 비교까지 한 번에 수행.

### 1.2 타겟
- **B2C:** 개인 투자자 — 기관 수준 구조를 단순한 UI로 제공.
- **B2B:** 애널리스트, PM, Corp Dev — 공시 → 인사이트 → 밸류를 한 흐름으로 처리.

### 1.3 단위 경제
- **정성:** Gemini **1회 호출** (Item 7 또는 1A당) → 토큰 비용 최소화.
- **정량:** yfinance / yahooquery **무료** → 숫자 정확도 확보, API 비용 없음.

---

## 2. 아키텍처 개요

### 2.1 하이브리드 분리 원칙
| 파이프라인 | 담당 | 데이터 소스 | 비고 |
|-----------|------|-------------|------|
| **정성 (Qualitative)** | LLM (Gemini 2.0 Flash) | SEC EDGAR 10-K (Item 1A, 3, 7, 9A) | 토큰 절감을 위해 Item 7·1A만 전송 |
| **정량 (Quantitative)** | Pandas + yfinance/yahooquery | Yahoo Finance API | TTM/분기 폴백, 다중 속성 폴백 |

### 2.2 아키텍처 다이어그램 (요약)
```
User → Streamlit UI (3 Tabs)
  ├─ Tab1: 10-K & MD&A Insights
  │    ├─ Qual: SEC 10-K → Parser → Item 1A/7 → Gemini → 전략·리스크 리포트
  │    └─ Quant: yfinance → DuPont, Altman Z, Sankey, Radar, F-Score, YoY, Red Flags
  ├─ Tab2: DCF Valuation
  │    └─ Quant: yfinance → FCF/Debt/Cash/Shares → 10Y 2-Stage DCF, Bull/Base/Bear
  └─ Tab3: Industry Analysis
       ├─ Quant: yfinance → Peer P/E, EV/EBITDA, P/B (조건부 포맷팅)
       └─ Qual: Gemini → Industry Outlook (거시 트렌드)
```

### 2.3 주요 아키텍처 결정 (ADL)
- **ADL-001:** 429 대응 — Item 7·1A만 추출, Item 8 숫자는 yfinance로 대체 → 토큰 80%+ 절감.
- **ADL-002:** 10년 2단계 DCF — 5년 DCF의 터미널 가치 왜곡 완화; Y1–5 성장, Y6–10 Fade 후 TV.
- **ADL-003:** 다중 폴백 — yahooquery 1순위 → yfinance (fast_info → info → balance_sheet) → TTM/분기 합산.

---

## 3. 기술 스택 및 인프라

### 3.1 런타임
- **Python:** 3.9+
- **프레임워크:** Streamlit ≥ 1.28.0

### 3.2 외부 API
| API | 라이브러리 | 용도 | 인증 |
|-----|-----------|------|------|
| Google Gemini | google-generativeai ≥ 0.8.0 | MD&A 인사이트, 리스크, 산업 전망 | GOOGLE_API_KEY |
| SEC EDGAR | sec-edgar-downloader ≥ 5.0.0 | 최신 10-K HTML 다운로드 | SEC_EDGAR_EMAIL (User-Agent) |
| Yahoo Finance | yfinance ≥ 0.2.40, yahooquery ≥ 2.2.0 | 검색, 재무제표, 멀티플, 컨센서스 | 없음 |

### 3.3 데이터 처리
- **Pandas** ≥ 2.0.0 — 재무 시계열, DCF, comps 테이블.
- **BeautifulSoup4** ≥ 4.12.0, **lxml** ≥ 4.9.0 — HTML 파싱·클렌징.
- **Plotly** ≥ 5.18.0 — Sankey, Radar, 5년 트렌드 라인 차트.

---

## 4. 데이터 흐름 상세

### 4.1 정성 데이터 흐름 (Qualitative Flow)
1. **다운로드:** `sec-edgar-downloader`로 티커당 최신 10-K HTML (임시 디렉터리).
2. **슬라이싱:** `_slice_html_items_1a_to_9a()` — 원문에서 Item 1A ~ Item 9A 구간만 문자열로 추출 (대용량 파일 회피).
3. **파싱·클렌징:** `extract_text_from_html()` → BeautifulSoup(lxml)으로 table/img/script 제거 → `find_item_section_generic()`로 Item 1A, 3, 7, 8, 9A 추출 → `clean_text_for_llm()` (태그·공백 정리).
4. **캐싱:** `data/{TICKER}_latest.json`에 item1a, item3, item7, item8, item9a 저장. 재실행 시 다운로드 생략.
5. **Chunking:** `smart_chunk(section, max_chars=10000)` — 긴 Item 7은 앞·뒤 비율로 압축.
6. **LLM:** Gemini `generate_content` (또는 `stream=True`) — 전략(Item 7), 리스크(Item 1A), 포렌식(Item 3·9A). 429 시 `_generate_with_retry()`로 60초 대기 후 재시도.

### 4.2 정량 데이터 흐름 (Quantitative Flow)
1. **검색:** 사이드바 — yahooquery `search(query)` → EQUITY/ETF만 필터, INDEX/MUTUALFUND 제외 → `[Exchange] Symbol - Name` 선택.
2. **티커 정규화:** `get_global_ticker(ticker, market)` — US 그대로, 한국 .KS/.KQ, 일본 .T, UK .L.
3. **재무 데이터:** `_get_annual_financials_balance_cashflow(ticker)` — yahooquery 1순위, 실패 시 yfinance annual → 비어 있으면 quarterly(TTM) 폴백.
4. **DCF 입력:** `get_dcf_inputs()` — FCF=OCF−CapEx, Shares (fast_info → info → balance_sheet), Debt/Cash 동일 다단계 폴백.
5. **시각화·지표:** DuPont, Altman Z, Piotroski F-Score, Sankey, Radar, YoY, Red Flags — 모두 `_safe_float()` 및 N/A 처리.

---

## 5. 코드 구조

### 5.1 파일 레이아웃
```
FQDC Project/
├── app.py              # 메인 앱 (~3039줄): UI, SEC, Gemini, DCF, Comps, 차트
├── find_toc.py         # 10-K HTML 목차(TOC) 탐색 유틸 — CLI 또는 URL 인자
├── push_to_github.sh   # 원격 푸시 스크립트
├── requirements.txt   # 의존성
├── .env.example       # GOOGLE_API_KEY, SEC_EDGAR_EMAIL 템플릿
├── .gitignore         # venv, .env, .app_prefs.json, data/
├── data/              # 런타임: 10-K JSON 캐시 (티커별)
├── .app_prefs.json    # 런타임: API 키·이메일 로컬 저장 (선택)
├── .agent/            # 에이전트/문서
│   ├── prd.md         # 제품 요구사항
│   ├── architecture.mermaid
│   ├── flows.md       # 정성/정량 흐름 설명
│   ├── directory_map.md
│   ├── adl.yaml       # 아키텍처 결정
│   ├── infra.yaml     # 인프라·의존성
│   ├── manifest.json  # 메타데이터
│   └── rules.md       # 개발 가드레일
├── TECHNICAL_NOTES.md # 429 대응·토큰 최적화 기술 노트
├── README.md
└── AGENT.md           # 에이전트 진입점
```

### 5.2 app.py 함수 그룹 (요약)
| 구간 | 역할 | 대표 함수 |
|------|------|-----------|
| 설정·유틸 | 프리퍼런스, 경로 | `_load_prefs`, `_save_prefs`, `_PREFS_PATH`, `_DATA_DIR` |
| 티커·시장 | 검색·접미사 | `get_global_ticker`, `infer_market_from_ticker`, `SECTORS` |
| SEC·10-K | 다운로드·파싱·캐시 | `_slice_html_items_1a_to_9a`, `extract_text_from_html`, `find_item_section_generic`, `download_and_extract_all_items`, `get_10k_sections`, `_load_10k_from_cache`, `_save_10k_to_cache` |
| Gemini | 모델·재시도·스트리밍 | `get_gemini_model`, `_generate_with_retry`, `_generate_stream`, `get_gemini_item7_strategy`, `get_gemini_item7_strategy_stream`, `get_gemini_item1a_risks`, `get_gemini_item1a_risks_stream`, `_gemini_forensic_audit`, `get_sec_financials_llm`, `get_industry_outlook` |
| 정량·재무 | yfinance/yyahooquery | `_get_annual_financials_balance_cashflow`, `_get_row_series`, `get_dcf_inputs`, `get_dcf_smart_defaults`, `get_analyst_consensus`, `get_sector_industry`, `get_5yr_financial_trend` |
| DCF | 10년 2단계 | `dcf_10y_2stage`, `excel_style_dcf`, `_damodaran_wacc_for_sector` |
| 지표·차트 | DuPont, Altman, F-Score, Sankey, Radar | `get_dupont_altman_redflags_yoy`, `get_piotroski_fscore`, `get_income_statement_sankey_data`, `_build_sankey_figure`, `_build_radar_figure`, `get_sector_specific_metrics` |
| Comps | 동종사 멀티플 | `get_comps_data` |
| UI | 탭·사이드바 | 2305줄~ `st.tabs`, Tab1/2/3 블록, 사이드바 검색·선택 |

### 5.3 캐싱 전략
- **Streamlit:** `@st.cache_data(ttl=300)` (5분) — `get_dcf_inputs`, `get_analyst_consensus`, `get_dupont_altman_redflags_yoy`, `get_comps_data`, `get_5yr_financial_trend`, `get_sector_industry`, `get_radar_metrics_normalized`, `get_piotroski_fscore`, `get_sector_specific_metrics`, `_get_annual_financials_balance_cashflow` 등.
- **Item8 LLM:** `@st.cache_data(ttl=3600)` — `get_sec_financials_llm` (1시간).
- **10-K 본문:** `data/{ticker}_latest.json` — 캐시 존재 시 `get_10k_sections()`에서 다운로드·파싱 생략.
- **상태:** `st.session_state` — ticker, market, google_api_key, sec_email, mda_strategy_result, mda_risk_result, company_search_options 등.

---

## 6. 탭별 기능 상세

### 6.1 Tab 1 — 10-K & MD&A Insights
- **상단:** Sector/Industry 뱃지 (get_sector_industry).
- **Financial Health:**
  - Sankey: 손익 흐름 (Item 8 LLM 추출 또는 yfinance).
  - Radar: ROE, Current Ratio, Asset Turnover, Equity Mult., Revenue YoY (정규화).
  - Piotroski F-Score (9점), Altman Z-Score, Red Flags (Current Ratio < 1.0, Interest Coverage < 1.5).
  - Sector-specific: Tech(Rule of 40, FCF margin, R&D%), Retail(재고회전율, 영업이익률), Financials(ROE, ROA).
  - YoY 비율 변화, 분기별 모멘텀 테이블 (녹색/빨간색 조건부 스타일).
- **Deep-Dive (AI):**
  - US 한정: "Analyze Management Strategy (MD&A)" → Item 7 스트리밍; "Analyze Risk Factors (Item 1A)" → Item 1A 스트리밍 + Item 3·9A 포렌식.
  - 한국/일본/UK: 현재 경고 메시지 (DART/EDINET/LSE Phase 2 예정).
  - 결과는 session_state에 저장 후 expander로 이전 분석 표시.

### 6.2 Tab 2 — 3-Scenario DCF Valuation
- **5년 트렌드:** Revenue, Net Income, Operating Margin %, FCF (YoY), Plotly 라인 차트.
- **DCF 입력:** Base FCF (OCF−CapEx 자동 또는 수동), Shares/Debt/Cash (자동 다단계 폴백, 실패 시만 수동).
- **슬라이더:** WACC, Terminal Growth (기본 2.5%), Projected FCF Growth (Y1–5). 기본값: `get_dcf_smart_defaults()` — Beta(CAPM), revenueGrowth/earningsGrowth 기반.
- **Reference 패널:** 애널리스트 컨센서스 (목표가, 추천, 성장률) + Damodaran (섹터 WACC, ERP 4.6%, Rf 4.2%, 링크).
- **출력:** Bull/Base/Bear (FCF 성장률 ±2%) 내재가치, 현재가 대비, 3시나리오 테이블.

### 6.3 Tab 3 — Industry Analysis & Comps
- **산업 선택:** SECTORS (Semiconductors & Hardware, Software & Cloud, Consumer Retail, Financial Services, Healthcare) → 해당 산업 Top 5 티커.
- **Comps 테이블:** Forward P/E, EV/EBITDA, P/B — 최소값 녹색, 최대값 빨간색 조건부 포맷; N/A 처리.
- **AI Industry Outlook:** Gemini로 12–18개월 거시 트렌드, 성장 동력, 리스크 요약.

---

## 7. 핵심 알고리즘

### 7.1 10년 2단계 DCF (`dcf_10y_2stage`)
- **Stage 1 (Y1–5):** FCF가 매년 `fcf_growth`로 성장.
- **Stage 2 (Y6–10):** 성장률이 `fcf_growth`에서 `term_growth`까지 선형 Fade (`fade = (t-6)/4`).
- **Terminal Value:** Y10 FCF × (1 + term_growth) / (wacc - term_growth), Y10 시점으로 할인.
- **Equity:** EV − Total Debt + Cash; 주당가치 = Equity / Shares.

### 7.2 DuPont 3단계
- ROE = NPM × Asset Turnover × Equity Multiplier (연도별).
- NPM = Net Income / Revenue, AT = Revenue / Total Assets, EM = Total Assets / Equity.

### 7.3 Altman Z-Score
- Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MC/TL) + 1.0×(Sales/TA). Safe > 2.99, Distress < 1.81.

### 7.4 Smart Defaults (DCF)
- WACC: Beta(기본 1.0), Rf 4%, MRP 5% → CAPM 근사.
- Terminal: 2.5% (Damodaran 스타일).
- FCF Growth: revenueGrowth 또는 earningsGrowth (예: 0.15 → 15%); 없으면 8%.

---

## 8. 에러 처리 및 개발 규칙 (rules.md)

- **예외·폴백:** 모든 금융 API 호출 try/except; 실패 시 빈 DataFrame 또는 0.0/None; 수치 파싱은 `_safe_float()` 사용.
- **토큰:** HTML 클렌징 후 `smart_chunk()`; 429 시 `_generate_with_retry()` (60초 대기).
- **캐싱:** 재무/분석 `@st.cache_data(ttl=300)`; 10-K는 `data/` JSON 영구 캐시.
- **표시:** NaN/None은 `_na()`로 "N/A" 통일.

---

## 9. 보안 및 환경

- **비밀:** `.env`에 GOOGLE_API_KEY, SEC_EDGAR_EMAIL (`.gitignore`).
- **로컬 저장:** "Remember API key & email" 선택 시 `.app_prefs.json` (역시 `.gitignore`).
- **SEC 정책:** User-Agent에 연락용 이메일 필수.

---

## 10. 제한 사항 및 로드맵

- **한국/일본/UK:** 10-K 대신 DART/EDINET/LSE 연동은 "Phase 2" 또는 "under development" 상태; US만 전체 정성 플로우 지원.
- **실행 시간:** Tab 1 첫 10-K 로드 20–60초, Gemini 스트리밍 5–10초; Tab 2·3는 yfinance만으로 수 초.
- **로드맵:** MVP → 상용화(B2C/B2B SaaS) 목표.

---

## 11. 개선 제안

1. **모듈 분리:** app.py를 `sec_edgar.py`, `gemini_analysis.py`, `dcf.py`, `comps.py`, `charts.py`, `ui_tabs.py` 등으로 나누면 유지보수·테스트 용이.
2. **단위 테스트:** DCF 공식, DuPont/Altman 계산, `_safe_float`/폴백 로직에 대한 pytest 추가.
3. **AGENT.md 경로:** 문서가 `agent/`가 아닌 `.agent/`에 있으므로 AGENT.md 내 링크를 `./.agent/`로 통일하거나 디렉터리명 정리.
4. **TECHNICAL_NOTES.md:** `prefilter_after_item7` 등 현재 코드와 다른 함수명이 문서에 있을 수 있음 — 코드 기준으로 문서 동기화 권장.

---

이 문서는 프로젝트 루트의 `PROJECT_ANALYSIS.md`로 저장되었으며, 에이전트·신규 개발자가 전체 구조와 규칙을 빠르게 파악하는 데 활용할 수 있습니다.
