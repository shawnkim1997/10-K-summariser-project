# AI 기반 10-K 재무 분석기

사용자가 종목 티커(예: AAPL)를 입력하면 SEC EDGAR에서 최신 10-K를 가져와, Item 7(MD&A)와 Item 8(Financial Statements)을 바탕으로 CFA 관점의 3문장 요약과 핵심 재무 지표(Revenue, Net Income, Operating Cash Flow)를 보여주는 웹 앱입니다.

## 기술 스택

- **UI**: Streamlit  
- **데이터 수집**: sec-edgar-downloader (SEC EDGAR)  
- **AI 분석**: Google Gemini 1.5 Pro (google-generativeai)  

## 필요한 것

- Python 3.9 이상  
- [Google API Key (Gemini)](https://aistudio.google.com/apikey)  
- SEC EDGAR 접속 시 사용할 이메일 주소 (실명/실제 메일 권장)  
- (선택) `.env` 파일: `GOOGLE_API_KEY`, `SEC_EDGAR_EMAIL` 설정 시 사이드바에 자동 반영  

---

## 실행 방법 (단계별)

**한 줄 요약:** 터미널에서 프로젝트 폴더로 간 뒤, 가상환경 켜고 `pip install -r requirements.txt` 한 다음 `streamlit run app.py` 입력하면 됩니다.

### 1단계: 터미널 열기

- **Mac**: `Spotlight(Cmd+Space)` → "터미널" 입력 후 실행  
- **Windows**: `Win + R` → `cmd` 입력 후 실행  

### 2단계: 프로젝트 폴더로 이동

```bash
cd "/Users/seonpil/Documents/FQDC Project"
```

(다른 위치에 프로젝트를 둔 경우 해당 폴더 경로로 바꿔 주세요.)

### 3단계: 가상환경 만들기 (권장)

한 번만 하면 됩니다.

```bash
python3 -m venv venv
```

### 4단계: 가상환경 켜기

**Mac / Linux:**

```bash
source venv/bin/activate
```

**Windows (명령 프롬프트):**

```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

프롬프트 앞에 `(venv)`가 보이면 성공입니다.

### 5단계: 패키지 설치

```bash
pip install -r requirements.txt
```

**Gemini 전용으로 새로 설치하는 경우 (기존 앤스로픽 제거 후):**

```bash
pip uninstall anthropic -y
pip install google-generativeai python-dotenv
# 또는 전체 재설치
pip install -r requirements.txt
```

인터넷이 필요하며, 1~2분 정도 걸릴 수 있습니다.

### 6단계: 앱 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 에서 앱이 실행됩니다.  
자동으로 안 열리면 브라우저 주소창에 `http://localhost:8501` 을 입력하세요.

### 7단계: 설정 및 분석

1. **왼쪽 사이드바**에서  
   - **Anthropic API Key**: [Anthropic 콘솔](https://console.anthropic.com)에서 발급한 키 입력  
   - **SEC EDGAR 이메일 주소**: 본인 이메일 입력 (SEC 정책 준수용)  
2. 메인 화면에서 **종목 티커** 입력 (예: `AAPL`, `MSFT`)  
3. **분석 실행** 버튼 클릭  
4. 10-K 다운로드 및 AI 분석이 끝나면,  
   - **CFA 관점 3문장 요약** (재무 건전성, 수익성, 리스크)  
   - **Revenue / Net Income / Operating Cash Flow 표**  
   를 확인할 수 있습니다.

---

## 종료 방법

터미널에서 `Ctrl + C` 를 누르면 앱이 종료됩니다.

---

## 폴더 구조

```
FQDC Project/
├── app.py              # Streamlit 앱 (메인, Gemini 1.5 Pro)
├── requirements.txt    # 필요한 라이브러리 목록
├── .env.example        # API 키 예시 (복사 후 .env 로 저장해 사용)
└── README.md           # 이 파일
```

## 문제 해결

- **"10-K 파일을 찾을 수 없습니다"**  
  - 티커가 정확한지 확인 (예: AAPL, MSFT)  
  - 인터넷 연결 확인  
  - SEC EDGAR 이메일을 입력했는지 확인  

- **"Anthropic API Key를 입력해 주세요"**  
  - 사이드바에서 API Key를 입력했는지 확인  

- **패키지 설치 오류**  
  - `pip install --upgrade pip` 후 다시 `pip install -r requirements.txt`  

- **한글 깨짐**  
  - 터미널/IDE 인코딩을 UTF-8로 설정해 보세요.  

---

## 라이선스 및 면책

이 프로젝트는 학습·포트폴리오 목적입니다.  
SEC 데이터 사용 시 [SEC 정책](https://www.sec.gov/os/webmaster-faq#code-support)을,  
AI 결과 활용 시 Anthropic 이용약관을 준수해 주세요.
