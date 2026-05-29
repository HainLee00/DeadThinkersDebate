"""
Stance Coordinator Agent — v5

선정된 3인의 사상가에게 A/B 딜레마 입장을 배정합니다.
반드시 2:1 또는 1:2 배분 (3:0 금지).
"""
from core.config import AGENT_MODELS
from core.llm_client import call_llm, extract_json

_SYSTEM_PROMPT = """\
당신은 "Dead Thinkers Debate"의 Stance Coordinator입니다.
선정된 3인의 역사적 사상가에게 철학적 딜레마의 입장을 배정합니다.

[역할]
1. 사용자 질문을 이분법 딜레마(A vs B)로 재구성합니다.
2. 각 사상가에게 A 또는 B를 배정합니다.

[딜레마 구조화 원칙]
- A와 B는 서로 실질적으로 대립해야 합니다.
- 두 선택지 모두 진지하게 주장 가능해야 합니다.
- 예: "이직해야 할까?" → A: 이직한다 / B: 현직을 유지한다

[입장 배정 원칙]
- 반드시 2:1 또는 1:2 배분만 허용 (A:3 또는 B:3인 3:0 배분은 절대 금지)
- 각 사상가의 역사적 사상·철학과 자연스럽게 일치해야 함
- 사상가 프로필에 명시된 primary_domains와 strongest_axis_weights를 참고하세요

[응답 형식 — 반드시 아래 JSON만 출력]
{
  "dilemma": {
    "core_question": "...",
    "choice_A": {"id": "A", "label": "..."},
    "choice_B": {"id": "B", "label": "..."}
  },
  "refined_topic": "직접 행위로 한 명을 희생시켜 다섯 명을 구하는 것이 도덕적으로 허용되는가",
  "panel_rationale": "세 인물이 이 딜레마에 선정된 이유 1~2문장",
  "assignments": [
    {
      "thinker_id": "사상가_id",
      "assigned_choice": "A",
      "choice_rationale": "이 사상가가 이 입장을 취하는 역사적·철학적 근거 1~2문장"
    }
  ]
}
"""


def _build_thinker_summary(thinker: dict) -> str:
    """LLM 프롬프트용 사상가 요약 텍스트."""
    domains = ", ".join(d["id"] for d in thinker.get("primary_domains", []))
    axes = ", ".join(thinker.get("strongest_axis_weights", {}).keys())
    return (
        f"- {thinker['name_ko']} ({thinker['name_en']}, {thinker['era']}, id: {thinker['thinker_id']})\n"
        f"  주요 분야: {domains}\n"
        f"  핵심 철학 긴장축: {axes}"
    )


async def run_stance_coordinator(
    user_question: str,
    selected_group: list[dict],
) -> dict:
    """
    딜레마 프레임 + 3인 입장 배정.

    반환:
    {
        "dilemma": {"core_question", "choice_A", "choice_B"},
        "refined_topic": str,
        "panel_rationale": str,
        "assignments": [{"thinker_id", "assigned_choice", "choice_rationale"}, ...]
    }
    """
    thinker_lines = "\n".join(_build_thinker_summary(t) for t in selected_group)
    user_prompt = (
        f"사용자 고민: {user_question}\n\n"
        f"선정된 사상가 3인:\n{thinker_lines}"
    )

    text, _ = await call_llm(
        model=AGENT_MODELS["stance_coordinator"],
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
        json_mode=True,
        max_tokens=800,
    )

    try:
        result = extract_json(text)
    except Exception:
        result = _fallback_assignments(selected_group)
        result["dilemma"] = {
            "core_question": user_question,
            "choice_A": {"id": "A", "label": "찬성"},
            "choice_B": {"id": "B", "label": "반대"},
        }
        result["refined_topic"] = user_question
        result["panel_rationale"] = ""
        return result

    # 3:0 배분 검증
    assignments = result.get("assignments", [])
    choices = [a.get("assigned_choice", "") for a in assignments]

    if len(set(choices)) < 2 or len(assignments) != 3:
        # 강제 보정: 마지막 사상가를 소수 쪽으로 뒤집음
        result["assignments"] = _force_balance(assignments, selected_group)

    return result


def _force_balance(assignments: list[dict], selected_group: list[dict]) -> list[dict]:
    """3:0 배분을 2:1로 강제 보정."""
    if not assignments:
        # 기본 배정
        return [
            {"thinker_id": selected_group[0]["thinker_id"], "assigned_choice": "A", "choice_rationale": ""},
            {"thinker_id": selected_group[1]["thinker_id"], "assigned_choice": "B", "choice_rationale": ""},
            {"thinker_id": selected_group[2]["thinker_id"], "assigned_choice": "A", "choice_rationale": ""},
        ]

    choices = [a.get("assigned_choice", "A") for a in assignments]
    if len(set(choices)) >= 2:
        return assignments

    # 모두 같은 입장 → 마지막을 뒤집음
    flip = "B" if choices[0] == "A" else "A"
    result = list(assignments)
    result[-1] = dict(result[-1])
    result[-1]["assigned_choice"] = flip
    return result


def _fallback_assignments(selected_group: list[dict]) -> dict:
    return {
        "assignments": [
            {"thinker_id": selected_group[0]["thinker_id"], "assigned_choice": "A", "choice_rationale": ""},
            {"thinker_id": selected_group[1]["thinker_id"], "assigned_choice": "B", "choice_rationale": ""},
            {"thinker_id": selected_group[2]["thinker_id"], "assigned_choice": "A", "choice_rationale": ""},
        ]
    }
