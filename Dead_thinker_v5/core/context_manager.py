import json

from core.config import AGENT_MODELS
from core.llm_client import call_llm, extract_json
from core.models import DebateSession, Round

_SUMMARIZER_SYSTEM = """\
아래 토론 발언들에서 각 사상가의 핵심 주장을 2~3문장으로 요약하세요.
Moderator 발언은 1~2문장으로 요약하세요.
사상가의 고유 논점만 남기고 수사적 표현은 생략하세요.

중요: thinker_summaries의 키는 발언 헤더에 있는 영문 id(예: sartre, nietzsche, mill)를 그대로 사용하세요.
한국어 이름이 아닌 영문 id를 키로 사용해야 합니다.

다음 JSON 형식으로만 응답하세요:
{
  "thinker_summaries": {
    "mill": "2~3문장 요약",
    "kant": "2~3문장 요약"
  },
  "moderator_summary": "1~2문장 요약"
}"""

_TOPIC_CHECKER_SYSTEM = """\
당신은 토론 주제 관련성을 판별합니다.
사용자 입력이 원래 토론 주제와 관련 있는지 판단하세요.
관련성을 넓게 해석하세요 — 감정적으로라도 연결되면 관련 있음.
완전히 무관한 주제 전환만 "관련 없음"으로 판별하세요.

JSON으로만 응답하세요 (다른 텍스트 없이):
{"is_relevant": true, "reason": "판별 근거 한 문장"}"""


async def summarize_round(round_data: Round) -> dict:
    """한 라운드의 Thinker/Moderator 응답을 LLM으로 요약. 결과를 Round.context_summary에 저장."""
    if round_data.context_summary is not None:
        return json.loads(round_data.context_summary)

    thinker_block = "\n\n".join(
        f"[{msg.speaker} (id: {msg.metadata.get('thinker_id', msg.speaker)})]\n{msg.content}"
        for msg in round_data.thinker_responses
    )
    mod_block = ""
    if round_data.moderator_response:
        mod_block = f"\n\n[Moderator]\n{round_data.moderator_response.content}"

    user_prompt = f"라운드 {round_data.round_num} 발언:\n\n{thinker_block}{mod_block}"

    text, _ = await call_llm(
        model=AGENT_MODELS["summarizer"],
        system_prompt=_SUMMARIZER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        json_mode=True,
        max_tokens=600,
    )

    try:
        result = extract_json(text)
    except Exception:
        result = {"thinker_summaries": {}, "moderator_summary": ""}

    round_data.context_summary = json.dumps(result, ensure_ascii=False)
    return result


async def build_r2plus_context(session: DebateSession, current_input: str) -> dict:
    """R2+ 컨텍스트를 구조화된 요약으로 구성."""
    refined_topic = session.rounds[0].gate_result["refined_topic"] if session.rounds else ""

    previous_rounds_summary = []
    for rd in session.rounds:
        summary = await summarize_round(rd)
        previous_rounds_summary.append({
            "round_num": rd.round_num,
            "user_input": rd.user_input,
            "thinker_summaries": summary.get("thinker_summaries", {}),
            "moderator_summary": summary.get("moderator_summary", ""),
        })

    return {
        "original_question": session.original_question,
        "refined_topic": refined_topic,
        "current_user_input": current_input,
        "round_num": len(session.rounds) + 1,
        "previous_rounds_summary": previous_rounds_summary,
    }


async def check_topic_relevance(original_topic: str, user_input: str) -> dict:
    """사용자의 R2+ 입력이 원래 토론 주제와 관련 있는지 판별."""
    text, _ = await call_llm(
        model=AGENT_MODELS["topic_checker"],
        system_prompt=_TOPIC_CHECKER_SYSTEM,
        user_prompt=f"원래 토론 주제: {original_topic}\n사용자 입력: {user_input}",
        temperature=0.0,
        json_mode=True,
        max_tokens=150,
    )

    try:
        return extract_json(text)
    except Exception:
        return {"is_relevant": True, "reason": "판별 실패, 관련 있음으로 처리"}
