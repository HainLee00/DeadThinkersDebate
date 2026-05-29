from pathlib import Path

from core.models import DebateSession

THINKER_EMOJI: dict[str, str] = {
    "mill":       "⚖️",
    "nozick":     "🗽",
    "aristotle":  "🏺",
    "marx":       "🔨",
    "rawls":      "⚖️",
    "nietzsche":  "⚡",
    "sartre":     "🎭",
    "beauvoir":   "✊",
    "kant":       "📐",
    "plato":      "💡",
}

_PASS_FAIL_LABEL = {
    "pass":       "✅ PASS",
    "borderline": "⚠️ BORDERLINE",
    "fail":       "❌ FAIL",
}


def write_selection_report(session_id: str, top3_groups: list[dict], output_dir: str) -> str:
    """사상가 선정 스코어링 결과를 마크다운으로 저장 (세션 파일과 분리)."""
    path = Path(output_dir) / f"selection_{session_id}.md"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Dead Thinkers Debate v5 — 사상가 선정 스코어",
        "",
        f"**세션 ID**: {session_id}",
        "",
        "| 순위 | 패널 | Group Score | Avg Individual | Coverage |",
        "|------|------|-------------|----------------|----------|",
    ]
    for i, grp in enumerate(top3_groups):
        rank = ["🥇 1위", "🥈 2위", "🥉 3위"][i]
        names = " / ".join(grp["name_ko"])
        lines.append(
            f"| {rank} | {names} "
            f"| {grp['group_score']:.4f} "
            f"| {grp['avg_individual_score']:.4f} "
            f"| {grp['coverage_score']:.4f} |"
        )

    lines += ["", "---", ""]

    for i, grp in enumerate(top3_groups):
        rank = ["🥇 1위 (선정)", "🥈 2위", "🥉 3위"][i]
        names = " / ".join(grp["name_ko"])
        lines += [
            f"## {rank} — {names}",
            "",
            f"- **Group Score**: {grp['group_score']:.4f}  "
            f"(= 0.65 × {grp['avg_individual_score']:.4f} + 0.35 × {grp['coverage_score']:.4f})",
            "",
            "**Individual Score (per thinker)**:",
        ]
        for tid, score in grp["individual_scores"].items():
            thinker_name = next(
                (grp["name_ko"][j] for j, t in enumerate(grp["thinker_ids"]) if t == tid),
                tid,
            )
            lines.append(f"- {thinker_name} (`{tid}`): {score:.4f}")
        lines += ["", "---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_session_report(session: DebateSession, output_dir: str) -> str:
    """대화 내용만 마크다운으로 저장. 경로 반환."""
    path = Path(output_dir) / f"session_{session.session_id}.md"

    gate_r1 = session.rounds[0].gate_result if session.rounds else {}
    refined_topic = (
        gate_r1.get("refined_topic", session.original_question)
        if gate_r1 else session.original_question
    )
    panel_rationale = gate_r1.get("panel_rationale", "-") if gate_r1 else "-"

    thinker_label = ", ".join(
        f"{THINKER_EMOJI.get(tid, '🧠')} {tid}" for tid in session.selected_thinkers
    )

    lines: list[str] = [
        "# Dead Thinkers Debate v5 — 대화 기록",
        "",
        f"**세션 ID**: {session.session_id}",
        f"**질문**: {session.original_question}",
        f"**정제된 주제**: {refined_topic}",
        f"**선정된 사상가**: {thinker_label}",
        f"**패널 선정 근거**: {panel_rationale}",
        "",
        "---",
        "",
    ]

    for rnd in session.rounds:
        lines += [f"## Round {rnd.round_num}", "", f"**사용자**: {rnd.user_input}", ""]

        for msg in rnd.thinker_responses:
            tid = msg.metadata.get("thinker_id", "")
            emoji = THINKER_EMOJI.get(tid, "🧠")
            lines += [f"### {emoji} {msg.speaker}", "", msg.content, ""]

        if rnd.moderator_response:
            lines += [
                "### 🎯 Moderator 종합",
                "",
                rnd.moderator_response.content,
                "",
            ]

        lines += ["---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_eval_report(eval_result: dict, session_id: str, output_dir: str) -> str:
    """평가 결과만 마크다운으로 저장. 경로 반환."""
    path = Path(output_dir) / f"eval_{session_id}.md"

    overall = eval_result.get("overall_score_10", "-")
    pass_fail_raw = eval_result.get("pass_fail", "-")
    pass_fail_label = _PASS_FAIL_LABEL.get(pass_fail_raw, pass_fail_raw)
    schema_ver = eval_result.get("evaluation_schema_version", "-")

    lines: list[str] = [
        "# Dead Thinkers Debate v5 — 평가 리포트",
        "",
        f"**세션 ID**: {session_id}",
        f"**평가 스키마**: {schema_ver}",
        f"**전체 점수**: {overall} / 10",
        f"**판정**: {pass_fail_label}",
        "",
        "---",
        "",
    ]

    core = eval_result.get("core_scores", {})
    if core:
        e1 = core.get("E1_perspective_separation", {})
        e2 = core.get("E2_reflection_elicitation", {})
        e3 = core.get("E3_persona_consistency", {})
        e4 = core.get("E4_safety", {})

        lines += [
            "## 핵심 평가 (E1~E4)",
            "",
            "| 기준 | 점수 |",
            "|------|------|",
            f"| E1. 관점 분리도 | {e1.get('score_1_to_5', '-')}/5 |",
            f"| E2. 성찰 유도성 | {e2.get('score_1_to_5', '-')}/5 |",
            f"| E3. 페르소나 일관성 | {e3.get('score_1_to_5', '-')}/5 |",
            f"| E4. 안전성 | {e4.get('score_1_to_5', '-')}/5 |",
            "",
        ]

        for label, key, criterion in [
            ("E1. 관점 분리도", "E1_perspective_separation", e1),
            ("E2. 성찰 유도성", "E2_reflection_elicitation", e2),
            ("E3. 페르소나 일관성", "E3_persona_consistency", e3),
            ("E4. 안전성", "E4_safety", e4),
        ]:
            score = criterion.get("score_1_to_5", "-")
            lines += [f"### {label} ({score}/5)", ""]

            rationale = criterion.get("rationale", "")
            if rationale:
                lines += [rationale, ""]

            evidence = criterion.get("evidence", [])
            if evidence:
                lines.append("**근거**:")
                for e in evidence:
                    lines.append(f"- {e}")
                lines.append("")

            issues = criterion.get("issues", [])
            if issues:
                lines.append("**개선점**:")
                for i in issues:
                    lines.append(f"- {i}")
                lines.append("")

            if key == "E3_persona_consistency":
                per_thinker = criterion.get("per_thinker", {})
                if per_thinker:
                    lines.append("**사상가별 평가**:")
                    for tid, note in per_thinker.items():
                        emoji = THINKER_EMOJI.get(tid, "🧠")
                        lines.append(f"- {emoji} **{tid}**: {note}")
                    lines.append("")

        lines += ["---", ""]

    diag = eval_result.get("auxiliary_diagnostics", {})
    if diag:
        d1 = diag.get("D1_conversational_naturalness", {})
        d2 = diag.get("D2_flow_appropriateness", {})
        d3 = diag.get("D3_user_continuability", {})
        d4 = diag.get("D4_debug_separation", {})

        lines += [
            "## 보조 진단 (D1~D4)",
            "",
            "| 진단 | 점수 | 근거 |",
            "|------|------|------|",
            f"| D1. 대화 자연스러움 | {d1.get('score_1_to_5', '-')}/5 | {d1.get('rationale', '-')} |",
            f"| D2. 흐름 적절성 | {d2.get('score_1_to_5', '-')}/5 | {d2.get('rationale', '-')} |",
            f"| D3. 사용자 계속성 | {d3.get('score_1_to_5', '-')}/5 | {d3.get('rationale', '-')} |",
            f"| D4. 디버그 분리 | {d4.get('score_1_to_5', '-')}/5 | {d4.get('rationale', '-')} |",
            "",
            "---",
            "",
        ]

    strengths = eval_result.get("strengths", [])
    major = eval_result.get("major_issues", [])
    minor = eval_result.get("minor_issues", [])
    fixes = eval_result.get("recommended_fixes", [])

    lines += ["## 종합 분석", ""]

    if strengths:
        lines.append("**강점**:")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    if major:
        lines.append("**주요 문제점**:")
        for m in major:
            lines.append(f"- {m}")
        lines.append("")

    if minor:
        lines.append("**소소한 문제점**:")
        for m in minor:
            lines.append(f"- {m}")
        lines.append("")

    if fixes:
        lines += ["---", "", "## 개선 제안", ""]
        lines += [
            "| 우선순위 | 대상 | 문제 | 제안 |",
            "|----------|------|------|------|",
        ]
        for fix in fixes:
            lines.append(
                f"| {fix.get('priority', '-')} | {fix.get('target', '-')} "
                f"| {fix.get('issue', '-')} | {fix.get('suggestion', '-')} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
