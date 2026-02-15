# All-in-One Financial Analysis Dashboard

**하이브리드 금융 대시보드** — 10-K·MD&A AI 인사이트, DCF/밸류에이션, 산업 비교, 포트폴리오·워치리스트, 크립토(빗썸/바이낸스)를 한 화면에서 제공합니다.

- **비용 효율**: 숫자/차트는 yfinance·YahooQuery 기반, **텍스트 인사이트만 Gemini** 사용
- **다중 시장**: 미국·한국 등 글로벌 티커 검색 및 통화(USD/KRW/JPY 등) 지원

---

## 최근 업데이트

### Portfolio & Watchlist
- **통화(Currency) 컬럼**: 종목별 매수 통화 선택 (USD, GBP, EUR, KRW, JPY, CNY). Sync 시 자산 거래 통화로 FX 환산 후 수익률 계산.
- **소수 수량(Quantity)**: 분수 주식·소수 주식 지원 (예: 30.395107). AI 스크린샷 추출 시에도 소수점 유지.
- **FX 반영 수익률**: 입력 평단가·통화 → 실시간 환율로 자산 통화 기준 `adjusted_avg` 계산 → `Total Return %` = (현재가 − adjusted_avg) / adjusted_avg. 다중 통화 포트폴리오에서 일관된 수익률 표시.
- **AI 스크린샷 (Gemini Vision)**: 브로커 화면에서 Ticker, Average Price, **Currency**($, £, €, ₩, ¥ → USD/GBP/EUR/KRW/JPY), **Quantity**(소수 포함) 자동 추출.

### 앱 전반 다중 통화·환율
- **티커별 통화**: `get_currency_for_ticker`로 종목 거래 통화 자동 감지 (USD/KRW/JPY 등).
- **실시간 환율**: `get_fx_rate`(통화→통화), `get_fx_rate_to_usd`(→USD). yfinance FX 페어 사용, TTL 60~120초 캐시.
- **가격 표기**: 비-USD 종목에 대해 `format_price_with_usd`로 로컬 통화 + USD 환산 병기 (예: ₩ 181,200 (≈ $ 132.50)).

### Valuation Hub (DCF / RIM)
- DCF·Reverse DCF·RIM 결과가 **티커 통화** 기준으로 표시되며, 비-USD일 경우 USD 환산액을 함께 표시.

### Earnings & Estimates
- 애널리스트 목표가·추천·현재가에 **티커 통화** 적용 및 필요 시 USD 환산 표기.

---

## 주요 기능 (8개 탭)

| 탭 | 설명 |
|----|------|
| **10-K & MD&A Insights** | SEC 10-K Item 7·Item 1A 기반 Gemini 요약, DuPont·Altman Z·Red Flags·YoY, Sankey·Radar·5년 재무·KPI |
| **Market Heatmap** | 섹터·매크로 히트맵, 금리·원유·VIX 등 지표 |
| **Valuation Hub** | 표준 DCF, Reverse DCF, RIM(잔여이익) 밸류에이션 |
| **Industry Analysis & Comps** | 산업 동료사 비교(Forward P/E, EV/EBITDA, P/B), AI 산업 전망 |
| **SEC Filings (Raw)** | SEC 제출 문서 목록 및 원문 링크 |
| **Earnings & Estimates** | 실적 컨센서스, Beat/Miss 차트, 애널리스트 목표가·추천 |
| **Portfolio & Watchlist** | 보유 종목 **종목별 통화(USD/GBP/EUR/KRW/JPY/CNY)**·**소수 수량** 지원, AI 스크린샷 추출, **실시간 FX 환산 수익률** |
| **Crypto** | 빗썸(KRW)·바이낸스(USD) 실시간 시세 |

---

## 설치 및 실행

### 요구 사항

- Python 3.10+
- [requirements.txt](requirements.txt) 의존성

### 1. 저장소 클론 후 의존성 설치

```bash
cd ici
pip install -r requirements.txt
```

### 2. 환경 변수 (선택)

- **Google API Key**: 사이드바에서 입력하거나 `.env`에 `GOOGLE_API_KEY=...` 설정  
  - Tab 1(10-K 인사이트), 포트폴리오 스크린샷 AI 추출에 사용
- **SEC 이메일**: 10-K 다운로드 시 User-Agent용 이메일 (사이드바 입력 권장)

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

---

## 프로젝트 구조

```
ici/
├── app.py              # 메인 Streamlit 앱 (탭·API·UI)
├── requirements.txt    # Python 의존성
├── data/               # 10-K 캐시 (ticker_latest.json 등, .gitignore 권장)
└── .app_prefs.json     # 로컬 저장 API 키·이메일 (선택, .gitignore 권장)
```

---

## 라이선스

이 프로젝트는 개인/교육 목적으로 제공됩니다. 외부 API(Google Gemini, SEC EDGAR, Yahoo Finance 등) 사용 시 각 서비스 약관을 확인하세요.
