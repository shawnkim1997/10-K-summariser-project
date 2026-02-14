# Agent Navigation Hub (AGENT.md)

이 문서는 AI 코딩 에이전트 및 개발자가 `10-K-summariser-project`의 구조와 컨텍스트를 빠르게 파악하기 위한 **진입점(Entrypoint)**입니다. 

작업을 시작하거나 코드를 수정하기 전에, 필요한 정보에 맞춰 아래의 문서를 먼저 확인하십시오. (모든 문서는 `agent/` 디렉토리에 위치합니다.)

## 🧭 Context & Documentation Map

| 문서명 | 역할 및 포함 내용 |
| :--- | :--- |
| **[PRD](./prd.md)** | 프로젝트 비전, 주요 기능 요구사항(3개의 탭), 타겟 유저 등 **프로젝트 기획 배경** |
| **[Architecture](./architecture.mermaid)** | 시스템의 전체적인 구조를 보여주는 **Mermaid 아키텍처 다이어그램** |
| **[Data Flows](./data_flows.md)** | 정성 파이프라인(LLM)과 정량 파이프라인(Pandas)이 어떻게 나뉘어 동작하는지 설명하는 **데이터 흐름도** |
| **[Directory Map](./directory_map.md)** | 루트 디렉토리 및 주요 파일들(`app.py`, `find_toc.py` 등)의 역할과 **파일 구조** |
| **[ADL](./adl.yaml)** | 429 에러 해결, 10년 2단계 DCF 도입, 다중 폴백 구조 등 **주요 기술적 의사결정 기록** |
| **[Infra](./infra.yaml)** | Python 버전, Streamlit, Gemini API, yfinance 등 **의존성 및 인프라 환경** |
| **[Manifest](./manifest.json)** | 프로젝트 메타데이터 (이름, 버전, 사용 기술 스택 등) |
| **[Rules](./rules.md)** | ⚠️ 에러 핸들링, 토큰 최적화, 상태 관리 등 코드를 작성할 때 반드시 지켜야 할 **개발 가드레일 및 규칙** |

---
**💡 Agent Action Item:** 코드를 수정하거나 새로운 기능을 구현할 때, 반드시 **[Rules](./rules.md)**를 먼저 숙지하고, 기존 아키텍처를 훼손하지 않도록 **[Architecture](./architecture.mermaid)** 및 **[Data Flows](./data_flows.md)**와 일치하게 작업하십시오.