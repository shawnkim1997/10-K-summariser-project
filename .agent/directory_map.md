# Directory Map

```text
FQDC Project/
├── app.py                  # 메인 스트림릿 애플리케이션 로직 (UI, API 연동, 데이터 처리)
├── find_toc.py             # SEC EDGAR 문서를 크롤링하여 목차(TOC) 위치를 식별하는 유틸리티 스크립트
├── push_to_github.sh       # GitHub 원격 저장소 자동 커밋/푸시 스크립트
├── requirements.txt        # 의존성 패키지 (streamlit, google-generativeai, yfinance 등)
├── README.md               # 프로젝트 소개, 실행 방법 및 아키텍처 설명
├── TECHNICAL_NOTES.md      # 토큰 최적화 및 Rate Limit 대응 기술 문서 (ADL 기반)
├── .env.example            # 환경변수 템플릿 파일
├── .gitignore              # Git 버전 관리 제외 목록 (가상환경, 로컬 설정 파일 등)
├── data/                   # (런타임 생성) 추출된 10-K 항목별 JSON 캐시 저장 폴더
└── .app_prefs.json         # (런타임 생성) 사용자 환경설정(API Key, Email)을 임시 저장하는 파일