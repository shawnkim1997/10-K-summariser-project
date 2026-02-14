# Product Requirements Document (PRD)
## 프로젝트명: All-in-One Financial Analysis Dashboard

### 1. 프로젝트 비전 및 목표
- **목표:** 주식 리서치 과정의 비효율성(방대한 공시 자료, 분산된 밸류에이션 모델 등)을 단일 워크플로우로 통합하는 하이브리드 대시보드 구축.
- **비전:** 개인 투자자와 금융 전문가(애널리스트, 포트폴리오 매니저 등)를 대상으로 하는 B2C/B2B SaaS 형태의 상용화.

### 2. 핵심 가치 제안 (하이브리드 아키텍처)
- 언어 모델(Gemini)은 텍스트 중심의 정성적 분석에만 사용하여 토큰 비용을 최소화.
- 정량적 수치(DCF, 멀티플 등)는 무료 API(`yfinance`, `yahooquery`)에서 가져와 재무 데이터의 정확성 확보.

### 3. 주요 기능 (Tabs)
1. **10-K & MD&A Insights:** SEC EDGAR에서 10-K(Item 1A, Item 7)를 가져와 Gemini로 경영진 어조, 전략적 변화, 잠재적 리스크 분석.
2. **3-Scenario DCF Valuation:** 10년 2단계 DCF 모델 (1~5년 성장, 6~10년 Fade). WACC, Terminal Growth 슬라이더 지원 및 Bull/Base/Bear 시나리오별 내재가치 도출.
3. **Top-Down Sector Analysis:** 특정 산업군 선택 시 경쟁사들의 멀티플(P/E, EV/EBITDA, P/B) 비교 및 Gemini 기반 거시적 산업 전망 생성.

### 4. 핵심 UI/UX 요구사항
- Streamlit 기반의 3개 탭 구성.
- 사이드바를 통한 전역 설정 (Google API Key, SEC Email, 다국어 지원 기업 검색).
- 정량 차트 시각화: Sankey Diagram, Radar Chart, F-Score 등 (Plotly 사용).