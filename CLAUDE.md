# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 저장소 구조

이 저장소의 실제 코드는 모두 `BPD/` 아래에 있으며, 이는 (서브모듈이 아닌) 별도의 중첩된 git 저장소입니다. 아래 모든 명령은 `cd BPD`를 먼저 실행했다고 가정합니다.

`BPD`는 논문 *Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation* (ICML 2026)의 **Backward Propagation Detection (BPD)** 공식 구현체입니다. 멀티에이전트 시스템(MAS)의 통신을 부호가 있는 DAG(signed DAG)로 모델링하고, 외부 LLM으로 엣지에 점수를 매기며, 역전파(backward propagation)를 통해 악의적인 에이전트를 탐지하고, 탐지된 공격자를 격리시켜 그래프를 복구합니다.

## 셋업 및 명령어

```bash
cd BPD
pip install -r requirements.txt
cp .env.example .env   # 이후 .env에 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 입력
```

MMLU 테스트 parquet 데이터(예: https://huggingface.co/datasets/cais/mmlu )를 `BPD/MMLU/` 아래에, `config.yaml`의 `data_path`와 경로가 일치하도록 배치하세요 (예: `MMLU/college_chemistry/test-00000-of-00001.parquet`).

파이프라인 실행:

```bash
python run.py
```

이 저장소에는 별도의 빌드 단계, 린터, 테스트 스위트가 없습니다 — `run.py`가 유일한 진입점입니다. 동작은 CLI 플래그가 아니라 전적으로 `config.yaml`(아래 참고)과 `.env`로 제어됩니다.

## 아키텍처

파이프라인(`run.py` → `mas.runner.communicate_with_bpd`)은 질문마다 세 단계로 구성됩니다.

1. **공격 실행(Attacked run)** — `mas.runner.communicate()`가 설정된 MAS 구조를 실행하며, 이때 한 에이전트(`malicious_agent_id`)는 고의로 잘못된 답을 주장하도록 지시받습니다(`mas.prompts.wrong_choice`, ground-truth `answer` 컬럼에서 도출). 모든 에이전트 턴(`mas.graph.ChatTurn.run`)은 `mas.llm.ask()`를 통해 답변을 생성하고, 자신이 응답한 각 메시지에 대해 `mas.llm.get_edge_score()`(동의/비동의를 판단하는 LLM)를 통해 `{-1, 0, 1}` 범위의 엣지 점수를 생성합니다. 이 점수들은 `mas.graph.EdgeGraph`라는 부호 있는 인접 행렬을 채우며, `(round, agent_id) -> node_index`로 평탄화된 인덱싱 방식(`node_index()`)을 사용합니다.

2. **탐지(BPD)** — `mas.bpd.run_bpd()`가 논문의 핵심 알고리즘을 구현합니다.
   - `backward_propagate()` (Eq. 1): 마지막 라운드의 노드 점수를 최종 다수결 답변과의 일치 여부에 따라 +1/-1로 초기화한 뒤, 라운드를 거슬러 역전파하며 각 노드의 점수를 자신의 가중 outgoing 엣지들에 대한 평균으로 계산합니다.
   - `aggregate_agent_scores()`: 각 응답자(respondent)의 첫 라운드와 마지막 라운드 노드 점수를 평균냅니다.
   - `detect_malicious()` (Eq. 2): 다른 모든 에이전트와의 평균 쌍별 점수 편차가 `epsilon`(논문 기본값 1.5)을 초과하는 에이전트 중 편차가 가장 큰 에이전트를 악의적 에이전트로 판정합니다.

3. **복구(Repair)** — `communicate()`를 `repair_mode=True`와 `blocked_agents`에 탐지된 에이전트를 넣어 재실행합니다. 이때 해당 에이전트의 프롬프트는 정상(non-malicious) 상태로 되돌아가고, 그 에이전트의 outgoing 엣지는 모두 0으로 지워집니다(`_clear_blocked_edges`). 이렇게 해야 복구된 그래프에서 해당 에이전트가 다시 (오탐으로) 탐지되지 않습니다.

두 가지 MAS 토폴로지가 이 파이프라인을 `structure`(config의 `mas_structure`)를 통해 공유합니다.
- **hierarchy** (5+2+5): respondent 5명이 독립적으로 답변 → advisor 2명이 각 respondent의 답변을 모두 비평 → respondent들이 두 advisor의 피드백을 반영해 최종 요약 답변 생성.
- **flat** (5+5+5): 에이전트 5명이 독립적으로 답변 → 모든 에이전트가 서로의 답변을 비평(상호 peer suggestion) → 각 에이전트가 모든 peer 피드백을 반영해 최종 요약 답변 생성.

두 빌더 함수(`mas/runner.py`의 `communicate_hierarchy`, `communicate_flat`)는 각각 `ChatTurn`들의 라운드를 구성하고, `mas.graph.run_round()`를 통해 실행하여 `answers`와 `edges`를 채우며, `communicate()`의 `structure` 분기를 통해 호출됩니다.

에이전트 페르소나(Assistant 1~5, 각기 다른 추론 스타일)와 모든 프롬프트 템플릿(악의적/troublemaker 버전 포함)은 `mas/prompts.py`에 있습니다. `mas.llm`은 단일 OpenAI 호환 클라이언트(모듈 레벨 싱글톤, `init_llm()`으로 설정하며 `ask()` 호출 전에 반드시 실행되어야 함)를 보유하고, `record_path`로의 로깅과 엣지 점수 파싱(`parse_score`는 `'[score] x'` 형식의 응답을 기대함)을 담당합니다.

결과는 질문 단위로 `results_dir`(config.yaml에서 설정, 예: `results/chemistry/result_<id>.pkl`)에 피클로 저장되며, 이는 공격 실행과 복구 실행의 answers/edges 및 BPD 탐지 결과 전체를 포함합니다. 이와 별도로 전체 결과를 모은 `all_results.pkl`도 저장됩니다. 실행 로그는 `outputs/run.log`에 기록됩니다.

## 설정 (`config.yaml`)

| 키 | 의미 |
|-----|---------|
| `mas_structure` | `flat` 또는 `hierarchy` |
| `epsilon` | BPD 탐지 임계값 (논문 기본값 1.5) |
| `malicious_agent_id` | 공격 대상이 되는 respondent (1부터 시작하는 인덱스) |
| `num_respondents` / `num_advisors` | 토폴로지 크기 |
| `question_start` / `question_end` | 처리할 MMLU parquet의 행 인덱스 범위 |
| `model` | `OPENAI_MODEL` 환경 변수가 설정되면 그 값으로 대체됨 |
| `data_path` | MMLU parquet 파일 경로 |
| `results_dir` / `record_path` | 출력 위치 |
