# 리서치 ReAct (LangGraph + Serper)

Ouroboros 시드(`research_react_app/.ouroboros/seed.yaml`)에 맞춘 워크플로입니다.  
저장소 루트의 `daily_assistant_core.py`를 import 하므로 **루트에서** 의존성 설치 후 실행하세요.

## 실행

```bash
# 저장소 루트에서
pip install -r requirements.txt
streamlit run research_react_app/streamlit_research_agent.py
```

## 구성

- `research_workflow.py` — LangGraph 그래프, Serper·스텁 도구
- `streamlit_research_agent.py` — Streamlit UI
- `reports/seed_report.html` — 시드 정리 보고서
- `.ouroboros/seed.yaml` — 요구사항 시드
