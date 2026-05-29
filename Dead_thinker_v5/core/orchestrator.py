import asyncio
import json
import uuid
from pathlib import Path

from core.models import DebateSession, Message, Round, ThinkerProfile
from core.context_manager import build_r2plus_context, check_topic_relevance
from core.report_writer import write_session_report, write_eval_report, write_selection_report
from agents.selector_agent import run_selector
from agents.stance_coordinator import run_stance_coordinator
from agents.thinker_agent import run_thinker
from agents.moderator_agent import run_moderator, run_moderator_redirect
from agents.evaluator_agent import run_evaluator


# ─── Profile 빌더 ────────────────────────────────────────────────────────────

def _build_profile(thinker: dict, assignment: dict) -> ThinkerProfile:
    """YAML thinker dict + stance assignment → ThinkerProfile."""
    era = thinker.get("era", "")
    name_en = thinker.get("name_en", thinker.get("thinker_id", ""))
    identifier = f"{name_en} ({era})"

    return ThinkerProfile(
        id=thinker["thinker_id"],
        name=name_en,
        name_ko=thinker.get("name_ko", name_en),
        identifier=identifier,
        why_selected="",
        expected_stance="",
        assigned_choice=assignment.get("assigned_choice", ""),
        choice_rationale=assignment.get("choice_rationale", ""),
        response_guidance_ko=thinker.get("response_guidance_ko", {}),
    )


def _merge_figure(thinker: dict, assignment: dict) -> dict:
    """gate_result["selected_figures"] 저장용 — YAML dict + 입장 정보 병합."""
    return {
        "id": thinker["thinker_id"],
        "name": thinker.get("name_en", thinker["thinker_id"]),
        "name_ko": thinker.get("name_ko", thinker["thinker_id"]),
        "era": thinker.get("era", ""),
        "identifier": f"{thinker.get('name_en', '')} ({thinker.get('era', '')})",
        "primary_domains": thinker.get("primary_domains", []),
        "strongest_question_strengths": thinker.get("strongest_question_strengths", {}),
        "strongest_axis_weights": thinker.get("strongest_axis_weights", {}),
        "core_concepts": thinker.get("core_concepts", []),
        "response_guidance_ko": thinker.get("response_guidance_ko", {}),
        "assigned_choice": assignment.get("assigned_choice", ""),
        "choice_rationale": assignment.get("choice_rationale", ""),
    }


def _profiles_from_session(session: DebateSession) -> dict[str, ThinkerProfile]:
    """gate_result["selected_figures"]에서 ThinkerProfile 재구성."""
    figures = session.rounds[0].gate_result.get("selected_figures", [])
    profiles = {}
    for fig in figures:
        p = ThinkerProfile(
            id=fig["id"],
            name=fig.get("name", fig["id"]),
            name_ko=fig.get("name_ko", fig["id"]),
            identifier=fig.get("identifier", fig["id"]),
            assigned_choice=fig.get("assigned_choice", ""),
            choice_rationale=fig.get("choice_rationale", ""),
            response_guidance_ko=fig.get("response_guidance_ko", {}),
        )
        profiles[fig["id"]] = p
    return profiles


# ─── 직렬화 + 저장 ────────────────────────────────────────────────────────────

def _to_dict(obj) -> object:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _save_session(session: DebateSession, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dict = _to_dict(session)
    session_dict.pop("evaluation", None)
    path = output_dir / f"session_{session.session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_dict, f, ensure_ascii=False, indent=2)
    return path


def _save_eval(eval_result: dict, session_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    return path


async def evaluate_and_save(session: DebateSession, output_dir: str) -> dict:
    profiles = _profiles_from_session(session)
    eval_result = await run_evaluator(session, profiles)
    session.evaluation = eval_result

    out = Path(output_dir)
    _save_eval(eval_result, session.session_id, out)
    write_eval_report(eval_result, session.session_id, str(out))
    return eval_result


# ─── 공통 R1 로직 ────────────────────────────────────────────────────────────

async def _run_r1(user_input: str, output_dir: str = "outputs") -> dict:
    """
    Selector (Phase 1+2) → Stance Coordinator → Thinker×3 → Moderator.
    성공/실패 dict 반환.
    """
    session_id = uuid.uuid4().hex[:8]

    # Phase 1+2: YAML-based group selection
    sel = await run_selector(user_input)

    if not sel["is_relevant"]:
        return {"success": False, "redirect_message": sel["redirect_message"]}

    top3_groups: list[dict] = sel.get("top3_groups", [])

    # 선정 스코어 MD 파일 저장 (세션 파일과 분리)
    selection_report_path = write_selection_report(session_id, top3_groups, output_dir)

    selected_group: list[dict] = sel["selected_group"]

    # Stance Coordinator: dilemma framing + A/B assignment
    stance = await run_stance_coordinator(user_input, selected_group)

    assignments: list[dict] = stance.get("assignments", [])
    assignment_map = {a["thinker_id"]: a for a in assignments}

    # Build profiles (order: match assignment order)
    profiles: list[ThinkerProfile] = []
    selected_ids: list[str] = []
    for thinker in selected_group:
        tid = thinker["thinker_id"]
        asgn = assignment_map.get(tid, {"assigned_choice": "A", "choice_rationale": ""})
        profiles.append(_build_profile(thinker, asgn))
        selected_ids.append(tid)

    dilemma = stance.get("dilemma", {})
    refined_topic = stance.get("refined_topic", user_input)
    panel_rationale = stance.get("panel_rationale", "")

    session = DebateSession(
        session_id=session_id,
        original_question=user_input,
        selected_thinkers=selected_ids,
        rounds=[],
        is_ended=False,
    )

    thinker_context = {
        "original_question": user_input,
        "refined_topic": refined_topic,
        "current_user_input": user_input,
    }

    thinker_responses: list[Message] = list(await asyncio.gather(
        *[run_thinker(thinker_context, p, 1, other_thinkers_summary=None, dilemma=dilemma)
          for p in profiles]
    ))
    moderator_response = await run_moderator(thinker_context, thinker_responses, 1)

    # gate_result: session 재구성에 필요한 모든 정보 저장
    gate_result = {
        "selected_figures": [
            _merge_figure(t, assignment_map.get(t["thinker_id"], {}))
            for t in selected_group
        ],
        "selected_thinkers": selected_ids,
        "dilemma": dilemma,
        "refined_topic": refined_topic,
        "panel_rationale": panel_rationale,
        "question_scores": sel["question_scores"],
        "group_score": sel["group_score"],
    }

    session.rounds.append(Round(
        round_num=1,
        user_input=user_input,
        gate_result=gate_result,
        thinker_responses=thinker_responses,
        moderator_response=moderator_response,
        context_summary=None,
    ))

    return {
        "success": True,
        "session": session,
        "selector_result": sel,
        "stance": stance,
        "selected_ids": selected_ids,
        "refined_topic": refined_topic,
        "dilemma": dilemma,
        "figures": gate_result["selected_figures"],
        "thinker_responses": thinker_responses,
        "moderator_response": moderator_response,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

async def run_debate(output_dir: str = "outputs") -> None:
    output_path = Path(output_dir)

    print("\n" + "=" * 60)
    print("  Dead Thinkers Debate  v5  (Precision Selector)")
    print("=" * 60)
    print("\n어떤 인생의 고민이 있나요?\n")

    user_input = input("당신: ").strip()
    if not user_input:
        print("입력이 없어 종료합니다.")
        return

    print("\n[시스템] 질문에 맞는 인물을 찾는 중...\n")
    r1 = await _run_r1(user_input, output_dir)

    if not r1["success"]:
        print(f"\n[시스템] {r1['redirect_message']}")
        return

    session: DebateSession = r1["session"]
    refined_topic = r1["refined_topic"]

    names = ", ".join(fig["name_ko"] for fig in r1["figures"])
    print(f"[시스템] 선정된 인물: {names}")
    print(f"[시스템] 주제: {refined_topic}\n")

    for msg in r1["thinker_responses"]:
        print(f"\n[{msg.speaker}]\n{msg.content}")
    print(f"\n[{r1['moderator_response'].speaker}]\n{r1['moderator_response'].content}")

    round_num = 2

    while True:
        print()
        user_input = input("당신 (계속하거나 '종료' 입력): ").strip()

        if user_input in ("종료", "exit", "quit"):
            session.is_ended = True
            saved_path = _save_session(session, output_path)
            report_path = write_session_report(session, str(output_path))
            eval_result = await evaluate_and_save(session, str(output_path))

            print(f"\n[시스템] 대화 저장: {saved_path}")
            print(f"[시스템] 대화 리포트: {report_path}")
            print(f"\n[평가 결과]\n{json.dumps(eval_result, ensure_ascii=False, indent=2)}")
            print("\n[시스템] 대화를 마칩니다. 감사합니다.")
            break

        profiles_map = _profiles_from_session(session)
        dilemma = session.rounds[0].gate_result.get("dilemma", {})
        relevance = await check_topic_relevance(original_topic=refined_topic, user_input=user_input)

        if not relevance["is_relevant"]:
            mod_resp = await run_moderator_redirect(
                original_topic=refined_topic, user_input=user_input, round_num=round_num,
            )
            print(f"\n[{mod_resp.speaker}]\n{mod_resp.content}")
            session.rounds.append(Round(
                round_num=round_num, user_input=user_input,
                gate_result=None, thinker_responses=[], moderator_response=mod_resp,
                context_summary=None,
            ))
        else:
            await build_r2plus_context(session, user_input)

            last_round_content = {
                msg.metadata.get("thinker_id", ""): msg.content
                for msg in session.rounds[-1].thinker_responses
                if msg.metadata.get("thinker_id")
            }
            all_self_content: dict[str, list[str]] = {}
            for rd in session.rounds:
                for msg in rd.thinker_responses:
                    tid = msg.metadata.get("thinker_id", "")
                    if tid:
                        all_self_content.setdefault(tid, []).append(msg.content)

            thinker_context = {
                "original_question": session.original_question,
                "refined_topic": refined_topic,
                "current_user_input": user_input,
            }
            tasks = []
            for profile in profiles_map.values():
                other_quotes = [
                    {"name": profiles_map[oid].name_ko, "summary": last_round_content.get(oid, "")}
                    for oid in session.selected_thinkers
                    if oid != profile.id and last_round_content.get(oid)
                ]
                self_rounds = all_self_content.get(profile.id) or None
                tasks.append(run_thinker(
                    thinker_context, profile, round_num,
                    other_thinkers_summary=other_quotes or None,
                    self_previous_rounds=self_rounds,
                    dilemma=dilemma,
                ))

            thinker_responses = list(await asyncio.gather(*tasks))
            mod_resp = await run_moderator(thinker_context, thinker_responses, round_num)

            for msg in thinker_responses:
                print(f"\n[{msg.speaker}]\n{msg.content}")
            print(f"\n[{mod_resp.speaker}]\n{mod_resp.content}")

            session.rounds.append(Round(
                round_num=round_num, user_input=user_input,
                gate_result=None, thinker_responses=thinker_responses,
                moderator_response=mod_resp, context_summary=None,
            ))

        round_num += 1


# ─── Web API (Gradio용) ───────────────────────────────────────────────────────

async def run_r1_web(user_input: str, output_dir: str = "outputs") -> dict:
    return await _run_r1(user_input, output_dir)


async def run_r2plus_web(session: DebateSession, user_input: str) -> dict:
    refined_topic: str = session.rounds[0].gate_result["refined_topic"]
    dilemma = session.rounds[0].gate_result.get("dilemma", {})
    round_num = len(session.rounds) + 1
    profiles_map = _profiles_from_session(session)

    relevance = await check_topic_relevance(original_topic=refined_topic, user_input=user_input)

    if not relevance["is_relevant"]:
        mod_resp = await run_moderator_redirect(
            original_topic=refined_topic, user_input=user_input, round_num=round_num,
        )
        session.rounds.append(Round(
            round_num=round_num, user_input=user_input,
            gate_result=None, thinker_responses=[], moderator_response=mod_resp,
            context_summary=None,
        ))
        return {"is_redirect": True, "thinker_responses": [], "moderator_response": mod_resp}

    await build_r2plus_context(session, user_input)

    last_round_content = {
        msg.metadata.get("thinker_id", ""): msg.content
        for msg in session.rounds[-1].thinker_responses
        if msg.metadata.get("thinker_id")
    }
    all_self_content: dict[str, list[str]] = {}
    for rd in session.rounds:
        for msg in rd.thinker_responses:
            tid = msg.metadata.get("thinker_id", "")
            if tid:
                all_self_content.setdefault(tid, []).append(msg.content)

    thinker_context = {
        "original_question": session.original_question,
        "refined_topic": refined_topic,
        "current_user_input": user_input,
    }
    tasks = []
    for profile in profiles_map.values():
        other_quotes = [
            {"name": profiles_map[oid].name_ko, "summary": last_round_content.get(oid, "")}
            for oid in session.selected_thinkers
            if oid != profile.id and last_round_content.get(oid)
        ]
        self_rounds = all_self_content.get(profile.id) or None
        tasks.append(run_thinker(
            thinker_context, profile, round_num,
            other_thinkers_summary=other_quotes or None,
            self_previous_rounds=self_rounds,
            dilemma=dilemma,
        ))

    thinker_responses = list(await asyncio.gather(*tasks))
    mod_resp = await run_moderator(thinker_context, thinker_responses, round_num)

    session.rounds.append(Round(
        round_num=round_num, user_input=user_input,
        gate_result=None, thinker_responses=thinker_responses,
        moderator_response=mod_resp, context_summary=None,
    ))
    return {"is_redirect": False, "thinker_responses": thinker_responses, "moderator_response": mod_resp}


async def end_debate_web(session: DebateSession, output_dir: str) -> dict:
    session.is_ended = True
    output_path = Path(output_dir)
    session_path = _save_session(session, output_path)
    session_report = write_session_report(session, output_dir)
    eval_result = await evaluate_and_save(session, output_dir)

    return {
        "eval_result": eval_result,
        "session_path": str(session_path),
        "session_report": session_report,
        "eval_path": str(output_path / f"eval_{session.session_id}.json"),
        "eval_report": str(output_path / f"eval_{session.session_id}.md"),
    }
