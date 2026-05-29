"""
Thinker Agent — v5 (Precision Edition)

response_guidance_ko를 기반으로 사상가에 빙의하여 발화합니다.
주요 주장을 전면에 세우고 논거를 자연스럽게 전개합니다.
"""
from core.config import AGENT_MODELS
from core.llm_client import call_llm
from core.models import Message, ThinkerProfile


def _build_system_prompt(
    profile: ThinkerProfile,
    round_num: int,
    other_thinkers_summary: list[dict] | None = None,
    dilemma: dict | None = None,
    self_previous_rounds: list[str] | None = None,
) -> str:
    g = profile.response_guidance_ko

    prompt = f"""\
당신은 {profile.name_ko} ({profile.identifier})입니다.

[발화 지침]
기본 입장: {g.get('basic_position', '')}
전형적 논증 전략: {g.get('typical_move', '')}
주의사항: {g.get('caution', '')}

[공통 규칙]
- 한국어 존댓말로 답하세요.
- 번호나 개조식 없이 자연스러운 대화체 산문으로 답하세요.
- 당신의 역사적 배경과 사상을 바탕으로, 주요 주장을 전면에 내세우고 논거를 풀어가세요.
- 철학 용어나 전문 개념을 쓸 때는 간단히 풀어서 설명하세요.
- R1에서는 다른 인물을 직접 언급하지 마세요.
- 반드시 자신의 원전(저서·강연·서신 등)을 한 번 이상 직접 인용하세요.
  예: "제가 『실천이성비판』에서 썼듯이, ..."  /  "『논어』 위정편에서 저는 이렇게 말했습니다, ..."
- 청중이 공감할 수 있는 구체적인 예시(일상적 상황, 역사적 사례, 비유)를 하나 이상 들어주세요.\
"""

    if dilemma and profile.assigned_choice:
        choice_key = f"choice_{profile.assigned_choice}"
        choice_info = dilemma.get(choice_key, {})
        choice_label = choice_info.get("label", profile.assigned_choice)
        choice_A = dilemma.get("choice_A", {}).get("label", "A")
        choice_B = dilemma.get("choice_B", {}).get("label", "B")

        prompt += f"""

[딜레마]
{dilemma.get('core_question', '')}
- A: {choice_A}
- B: {choice_B}

당신의 배정된 입장: {profile.assigned_choice} — {choice_label}
입장 근거: {profile.choice_rationale}

[응답 형식 — 반드시 이 두 줄로 시작하세요]
선택: {choice_label}
주요논거: [당신의 핵심 논거 1~2문장]

이후 6~9문장으로 철학적 논거를 자연스럽게 전개하세요.
원전 인용과 구체적 예시가 자연스럽게 녹아들도록 하세요.\
"""
    else:
        prompt += """

- 5~8문장의 대화체 산문으로 자연스럽게 답하세요.
- 질문으로 끝낼 필요 없이 자신의 견해를 자연스럽게 마무리하면 됩니다.\
"""

    if round_num > 1 and other_thinkers_summary:
        other_blocks = "\n\n".join(
            f"[{t['name']}의 지난 발언]\n{t['summary']}"
            for t in other_thinkers_summary
        )
        prompt += f"""

[지난 라운드 — 다른 사상가들의 발언]
{other_blocks}

[R2+ 규칙 — 상대 반박]
- 다른 사상가의 발언 중 당신 입장과 가장 충돌하는 주장 하나를 직접 지목하고 반박하세요.
  예: "[이름]은 ~라고 하셨는데, 저는 ~라는 이유로 동의하기 어렵습니다."\
"""

    if round_num > 1 and self_previous_rounds:
        rounds_text = "\n\n".join(
            f"[{i + 1}라운드 발언]\n{content}"
            for i, content in enumerate(self_previous_rounds)
        )
        prompt += f"""

[당신의 이전 발언들 — 아래 표현·예시·인용구는 이번 라운드에 재사용 금지]
{rounds_text}

[R2+ 규칙 — 반복 금지]
- 위 발언에서 썼던 문장, 비유, 예시, 원전 인용구를 그대로 다시 쓰지 마세요.
- 입장(선택 A 또는 B)은 일관되게 유지하되, 논거는 완전히 새로운 각도에서 전개하세요.
- 자신의 사상 중 아직 꺼내지 않은 다른 측면, 다른 저작, 다른 개념을 활용하세요.
- 여전히 "선택: xxx\n주요논거: xxx" 형식으로 시작하세요.\
"""

    return prompt


def _build_user_prompt(
    original_question: str,
    refined_topic: str,
    current_user_input: str,
    round_num: int,
    dilemma: dict | None = None,
) -> str:
    if round_num == 1:
        if dilemma:
            core_q = dilemma.get("core_question", refined_topic)
            return (
                f"딜레마: {core_q}\n"
                f"원래 고민: {original_question}"
            )
        return (
            f"다음은 한 사람의 고민입니다: {refined_topic}\n"
            f"원문: {original_question}"
        )
    return f"사용자: {current_user_input}"


async def run_thinker(
    context: dict,
    profile: ThinkerProfile,
    round_num: int,
    other_thinkers_summary: list[dict] | None = None,
    self_previous_rounds: list[str] | None = None,
    dilemma: dict | None = None,
) -> Message:
    system_prompt = _build_system_prompt(
        profile, round_num, other_thinkers_summary, dilemma, self_previous_rounds
    )
    user_prompt = _build_user_prompt(
        original_question=context["original_question"],
        refined_topic=context["refined_topic"],
        current_user_input=context["current_user_input"],
        round_num=round_num,
        dilemma=dilemma,
    )

    text, meta = await call_llm(
        model=AGENT_MODELS["thinker"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=750,
    )

    return Message(
        role="thinker",
        speaker=profile.name_ko,
        content=text.strip(),
        round_num=round_num,
        metadata={
            "thinker_id": profile.id,
            "name": profile.name,
            "identifier": profile.identifier,
            "assigned_choice": profile.assigned_choice,
            "model_used": meta["model"],
            "usage": meta["usage"],
        },
    )
