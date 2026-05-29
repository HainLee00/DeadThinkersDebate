import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

_v5_env = Path(__file__).parent / "data" / ".env"
_v4_env = Path(__file__).parent.parent / "Dead_thinker_v4" / "data" / ".env"
load_dotenv(_v5_env if _v5_env.exists() else _v4_env)

import gradio as gr

from core.orchestrator import run_r1_web, run_r2plus_web, end_debate_web

_ROOT = Path(__file__).parent
_OUTPUT_DIR = str(_ROOT / "outputs")

_FIGURE_EMOJI: dict[str, str] = {
    "mill":      "⚖️",
    "nozick":    "🗽",
    "aristotle": "🏺",
    "marx":      "🔨",
    "rawls":     "⚖️",
    "nietzsche": "⚡",
    "sartre":    "🎭",
    "beauvoir":  "✊",
    "kant":      "📐",
    "plato":     "💡",
}

_PASS_FAIL_LABEL = {
    "pass":       "✅ PASS",
    "borderline": "⚠️ BORDERLINE",
    "fail":       "❌ FAIL",
}


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _emoji(figure_id: str) -> str:
    return _FIGURE_EMOJI.get(figure_id, "🧠")


def _fmt_eval(ev: dict) -> str:
    if not ev or "core_scores" not in ev:
        return str(ev)

    core = ev["core_scores"]
    e1 = core.get("E1_perspective_separation", {})
    e2 = core.get("E2_reflection_elicitation", {})
    e3 = core.get("E3_persona_consistency", {})
    e4 = core.get("E4_safety", {})

    overall = ev.get("overall_score_10", "-")
    pass_fail = _PASS_FAIL_LABEL.get(ev.get("pass_fail", ""), ev.get("pass_fail", "-"))

    header = f"**전체 점수**: {overall} / 10 &nbsp;&nbsp; **판정**: {pass_fail}\n\n"

    rows = (
        "| 기준 | 점수 | 채점 근거 |\n"
        "|------|------|----------|\n"
        f"| E1. 관점 분리도 | {e1.get('score_1_to_5', '-')}/5 | {e1.get('rationale', '-')} |\n"
        f"| E2. 성찰 유도성 | {e2.get('score_1_to_5', '-')}/5 | {e2.get('rationale', '-')} |\n"
        f"| E3. 페르소나 일관성 | {e3.get('score_1_to_5', '-')}/5 | {e3.get('rationale', '-')} |\n"
        f"| E4. 안전성 | {e4.get('score_1_to_5', '-')}/5 | {e4.get('rationale', '-')} |"
    )

    per = e3.get("per_thinker", {})
    per_lines = ""
    if per:
        per_lines = "\n\n**인물별 페르소나 평가**:\n" + "\n".join(
            f"- {_emoji(tid)} `{tid}`: {note}" for tid, note in per.items()
        )

    strengths = ev.get("strengths", [])
    major = ev.get("major_issues", [])
    extras = ""
    if strengths:
        extras += "\n\n**강점**: " + " / ".join(strengths)
    if major:
        extras += "\n\n**주요 문제점**: " + " / ".join(major)

    return header + rows + per_lines + extras


async def chat_handler(user_message: str, history: list, session_state):
    if not user_message.strip():
        return history, session_state, ""

    history = history + [_msg("user", user_message)]

    # ── 종료 ──────────────────────────────────────────────────────────────────
    if user_message.strip().lower() in ("종료", "exit", "quit"):
        if session_state is None:
            history.append(_msg("assistant", "진행 중인 대화가 없습니다."))
            return history, None, ""

        result = await end_debate_web(session_state, _OUTPUT_DIR)
        ev_text = _fmt_eval(result["eval_result"])
        history.append(_msg("assistant",
            f"### 📊 평가 결과\n\n{ev_text}\n\n"
            f"---\n💾 대화: `{result['session_path']}`  \n"
            f"📋 평가: `{result['eval_path']}`"
        ))
        return history, None, ""

    # ── R1 (첫 메시지) ────────────────────────────────────────────────────────
    if session_state is None:
        history.append(_msg("assistant", "⏳ 질문에 맞는 역사적 인물을 찾는 중입니다..."))

        result = await run_r1_web(user_message, _OUTPUT_DIR)
        history.pop()  # 로딩 메시지 제거

        if not result["success"]:
            history.append(_msg("assistant", f"⚠️ {result['redirect_message']}"))
            return history, None, ""

        figures = result["figures"]
        dilemma = result.get("dilemma", {})
        stance = result.get("stance", {})
        choice_A = dilemma.get("choice_A", {})
        choice_B = dilemma.get("choice_B", {})

        # 상위 3개 선정 스코어 블록
        top3: list[dict] = result.get("selector_result", {}).get("top3_groups", [])
        if top3:
            rank_labels = ["🥇 1위", "🥈 2위", "🥉 3위"]
            score_rows = "\n".join(
                f"| {rank_labels[i]} | {' / '.join(g['name_ko'])} "
                f"| {g['group_score']:.4f} | {g['avg_individual_score']:.4f} | {g['coverage_score']:.4f} |"
                for i, g in enumerate(top3)
            )
            top3_block = (
                "<details><summary>📊 사상가 선정 스코어 (Top 3 조합)</summary>\n\n"
                "| 순위 | 패널 | Group Score | Avg Individual | Coverage |\n"
                "|------|------|-------------|----------------|----------|\n"
                f"{score_rows}\n\n"
                "</details>\n\n"
            )
        else:
            top3_block = ""

        # 딜레마 블록
        dilemma_block = (
            f"**선택의 딜레마**: {dilemma.get('core_question', result['refined_topic'])}\n\n"
            f"- **A** {choice_A.get('label', '')}\n"
            f"- **B** {choice_B.get('label', '')}"
        )

        # 인물 소개 카드 (입장 포함)
        figure_cards = "\n".join(
            f"- {_emoji(fig['id'])} **{fig['name_ko']}** — **{fig.get('assigned_choice', '?')}**  \n"
            f"  {fig['identifier']}  \n"
            f"  *{fig.get('choice_rationale', '')}*"
            for fig in figures
        )

        intro = (
            f"{top3_block}"
            f"{dilemma_block}\n\n"
            f"**패널**:\n{figure_cards}\n\n"
            f"*{stance.get('panel_rationale', '')}*"
        )

        thinker_blocks = "\n\n---\n\n".join(
            f"### {_emoji(msg.metadata.get('thinker_id', ''))} {msg.speaker}\n\n{msg.content}"
            for msg in result["thinker_responses"]
        )
        mod_block = f"### 🎯 Moderator\n\n{result['moderator_response'].content}"

        history.append(_msg("assistant",
            f"{intro}\n\n---\n\n{thinker_blocks}\n\n---\n\n{mod_block}"
        ))
        return history, result["session"], ""

    # ── R2+ ───────────────────────────────────────────────────────────────────
    result = await run_r2plus_web(session_state, user_message)

    if result["is_redirect"]:
        history.append(_msg("assistant",
            f"### 🎯 Moderator\n\n{result['moderator_response'].content}"
        ))
    else:
        thinker_blocks = "\n\n---\n\n".join(
            f"### {_emoji(msg.metadata.get('thinker_id', ''))} {msg.speaker}\n\n{msg.content}"
            for msg in result["thinker_responses"]
        )
        mod_block = f"### 🎯 Moderator\n\n{result['moderator_response'].content}"
        history.append(_msg("assistant",
            f"{thinker_blocks}\n\n---\n\n{mod_block}"
        ))

    return history, session_state, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Dead Thinkers Debate v5") as app:
    gr.Markdown(
        "# 🧠 Dead Thinkers Debate — v5 (Precision Selector)\n"
        "*YAML 기반 정밀 매칭으로 최적의 역사적 사상가 3인을 선정합니다*\n\n"
        "인생의 고민이나 철학적 질문을 입력하면, 질문의 철학적 긴장축을 분석하여 "
        "가장 적합한 사상가 3인을 배정합니다. 대화를 마치려면 **종료**를 입력하세요."
    )

    session_state = gr.State(None)

    chatbot = gr.Chatbot(
        value=[_msg("assistant", "안녕하세요! 어떤 인생의 고민이 있으신가요?")],
        height=660,
        show_label=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="고민을 입력하세요... (종료: '종료' 입력)",
            show_label=False,
            scale=9,
        )
        send_btn = gr.Button("전송", scale=1, variant="primary")

    msg.submit(chat_handler, [msg, chatbot, session_state], [chatbot, session_state, msg])
    send_btn.click(chat_handler, [msg, chatbot, session_state], [chatbot, session_state, msg])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,   # v2:7860, v3:7861, v4:7862, v5:7863
        share=False,
        theme=gr.themes.Soft(),
    )
