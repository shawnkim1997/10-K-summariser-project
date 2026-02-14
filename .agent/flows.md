# Flows

본 애플리케이션은 **정성적 흐름(Qualitative Flow)**과 **정량적 흐름(Quantitative Flow)**을 완전히 분리하여 설계되었습니다.

## 1. 정성 데이터 흐름 (Qualitative Flow)
1. **다운로드:** `sec-edgar-downloader`를 사용하여 입력받은 티커와 이메일로 가장 최신 10-K HTML 문서를 가져옵니다.
2. **파싱 및 클렌징:** HTML 문서에서 `lxml`과 `BeautifulSoup`을 사용해 테이블, 이미지, 스크립트를 제거하고 정규식으로 Item 1A(Risk Factors)와 Item 7(MD&A) 섹션만 추출합니다.
3. **캐싱 및 Chunking:** 추출된 텍스트를 로컬 디렉토리(`data/`)에 캐시로 저장하고, Gemini API 한도를 넘지 않도록 `smart_chunk()` 함수를 통해 중간 내용을 생략하여 압축합니다.
4. **LLM 추론:** Gemini API를 호출하여 경영진 전략, 주요 리스크, 핵심 인사이트를 생성하고 결과를 화면에 스트리밍합니다.

## 2. 정량 데이터 흐름 (Quantitative Flow)
1. **검색 및 식별:** 사용자가 입력한 기업명으로 `yahooquery`를 통해 티커 및 거래소 식별자를 추론합니다.
2. **데이터 페칭:** `yfinance` 및 `yahooquery`를 통해 재무상태표, 손익계산서, 현금흐름표를 호출합니다.
3. **폴백(Fallback) 연산:** 특정 값이 없을 경우 `fast_info`, `info`, `quarterly` 데이터 순으로 TTM(Trailing 12 Months) 값을 대체 연산합니다.
4. **모델링 및 렌더링:** 전처리된 데이터를 바탕으로 Pandas 연산을 통해 DCF 내재가치, Piotroski F-Score, DuPont 분석 값을 산출하고 Plotly 차트(Radar, Sankey)로 시각화합니다.