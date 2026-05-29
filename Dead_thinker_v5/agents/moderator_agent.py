from core.config import AGENT_MODELS
from core.llm_client import call_llm
from core.models import Message

_SYSTEM_PROMPT = """\
당신은 "Dead Thinkers Debate"의 Moderator입니다.
세 사상가의 발언을 듣고 사용자와 자연스럽게 대화하는 역할입니다.

[핵심 원칙]
- 당신은 "정답"을 제시하지 않습니다.
- 사유의 계기를 만드는 것이 목표입니다.
- 사상가들의 입장을 왜곡하지 않고 공정하게 정리합니다.
- 세 사상가는 이미 발언했으므로, 당신만이 사용자에게 질문을 던집니다.

[응답 방식]
- 번호, 소제목, 개조식 없이 자연스러운 대화체 산문으로만 씁니다.
- 세 사상가의 공통점과 대립 지점을 자연스럽게 엮어서 흐름 있게 서술합니다.
- 마지막에 반드시 질문 하나를 던지세요.
  - "어떤 관점이 더 와닿으세요?" 같은 단순 선택 질문은 피하세요.
  - 사용자의 전제나 가치관 자체를 건드리는 깊은 질문이어야 합니다.
- 전체 8~12문장, 한국어, 사상가 이름은 한국어로.
- 특정 사상가 두둔 금지.
- 사용자의 원래 고민과 연결 지어 정리하세요.
- 응답 초반에 각 사상가의 선택 분포를 자연스럽게 언급하세요.
  예: "두 분은 B를 선택하셨고, 한 분은 A를 선택하셨네요."
- 마지막 줄: "(계속 대화하려면 생각을 나눠주세요. 끝내려면 '종료'를 입력하세요.)"
"""


def _format_responses(thinker_responses: list[Message]) -> str:
    parts = []
    for msg in thinker_responses:
        parts.append(f"[{msg.speaker}]\n{msg.content}")
    return "\n\n".join(parts)


def build_moderator_user_prompt(
    original_question: str,
    refined_topic: str,
    thinker_responses: list[Message],
    round_num: int,
    user_input: str,
) -> str:
    responses_block = _format_responses(thinker_responses)
    if round_num == 1:
        return (
            f"사용자의 고민: {refined_topic}\n\n"
            f"세 사상가의 응답:\n\n{responses_block}"
        )
    return (
        f"사용자의 원래 고민: {refined_topic}\n"
        f"이번 라운드에서 사용자가 한 말: {user_input}\n\n"
        f"세 사상가의 응답:\n\n{responses_block}"
    )


_REDIRECT_SYSTEM = """\
당신은 "Dead Thinkers Debate"의 Moderator입니다.
사용자가 원래 토론 주제에서 벗어난 발언을 했습니다.

[규칙]
- '주제에서 벗어났다'고 직접 말하지 마세요.
- 사용자의 발언에서 원래 주제와 연결될 수 있는 실마리를 찾아 활용하세요.
- 번호나 개조식 없이, 3~5문장의 대화체 산문으로 자연스럽게 답하세요.
- 마지막에 원래 주제로 되돌아오는 질문을 하나 던지세요.
- 한국어로 답하세요.
"""


async def run_moderator_redirect(
    original_topic: str,
    user_input: str,
    round_num: int,
) -> Message:
    user_prompt = (
        f"원래 토론 주제: {original_topic}\n"
        f"사용자 발언: {user_input}"
    )
    text, meta = await call_llm(
        model=AGENT_MODELS["moderator"],
        system_prompt=_REDIRECT_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.5,
        max_tokens=300,
    )
    return Message(
        role="moderator",
        speaker="Moderator",
        content=text.strip(),
        round_num=round_num,
        metadata={
            "model_used": meta["model"],
            "usage": meta["usage"],
            "is_redirect": True,
        },
    )


async def run_moderator(
    context: dict,
    thinker_responses: list[Message],
    round_num: int,
) -> Message:
    user_prompt = build_moderator_user_prompt(
        original_question=context["original_question"],
        refined_topic=context["refined_topic"],
        thinker_responses=thinker_responses,
        round_num=round_num,
        user_input=context["current_user_input"],
    )

    text, meta = await call_llm(
        model=AGENT_MODELS["moderator"],
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.45,
        max_tokens=600,
    )

    return Message(
        role="moderator",
        speaker="Moderator",
        content=text.strip(),
        round_num=round_num,
        metadata={
            "model_used": meta["model"],
            "usage": meta["usage"],
        },
    )
