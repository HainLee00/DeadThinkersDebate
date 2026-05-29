"""
Selector Agent — v5 (Precision Edition)

Two-phase selection:
  Phase 1 — LLM scores the user question against selection_options.yaml
  Phase 2 — Python computes individual/group scores from thinker_pool.yaml
             and selects the best group of 3 via C(10,3)=120 enumeration.
"""
from itertools import combinations
from pathlib import Path

import yaml

from core.config import AGENT_MODELS
from core.llm_client import call_llm, extract_json

# ─── Load YAML data at module startup ────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"

with open(_DATA_DIR / "selection_options.yaml", encoding="utf-8") as _f:
    _OPTIONS_YAML = yaml.safe_load(_f)

with open(_DATA_DIR / "thinker_pool.yaml", encoding="utf-8") as _f:
    _POOL_YAML = yaml.safe_load(_f)

_THINKERS: list[dict] = _POOL_YAML["thinkers"]
_THINKER_MAP: dict[str, dict] = {t["thinker_id"]: t for t in _THINKERS}

# ─── Build options listing text for LLM prompt ───────────────────────────────

def _build_options_text() -> str:
    cats = _OPTIONS_YAML["selection_categories"]
    lines: list[str] = []
    for cat_id, cat_data in cats.items():
        lines.append(f"\n[{cat_id}]  # {cat_data['description_ko']}")
        for opt in cat_data["options"]:
            lines.append(f"  {opt['id']}: {opt['label_ko']}")
    return "\n".join(lines)

_OPTIONS_TEXT = _build_options_text()

# ─── Phase 1: LLM question scoring ──────────────────────────────────────────

_SYSTEM_PROMPT = f"""\
당신은 철학적 질문 분석 에이전트입니다.
사용자의 고민이나 질문을 아래 4개 카테고리의 옵션들에 대해 0~1 사이의 관련성 점수로 분석합니다.

[규칙]
1. 점수 > 0인 옵션만 JSON에 포함하세요 (없는 항목 = 0점으로 처리됨)
2. 각 카테고리에서 가장 관련 높은 옵션은 1.0에 가깝게, 덜 관련된 옵션은 낮게 분산 배분하세요
3. 완전히 무관한 질문(단순 사실 조회, 수학 계산, 유해 내용)은 is_relevant: false로 표시하세요

[옵션 목록]
{_OPTIONS_TEXT}

[출력 JSON 형식 — 다른 텍스트 없이 JSON만]
{{
  "is_relevant": true,
  "question_scores": {{
    "primary_domains": {{"option_id": score, ...}},
    "strongest_question_strengths": {{"option_id": score, ...}},
    "strongest_axis_weights": {{"option_id": score, ...}},
    "core_concepts": {{"option_id": score, ...}}
  }},
  "redirect_message": ""
}}

is_relevant = false인 경우:
{{
  "is_relevant": false,
  "question_scores": {{}},
  "redirect_message": "사용자에게 보여줄 안내 메시지 (한국어, 2~3문장)"
}}
"""


async def _score_question(user_question: str) -> dict:
    """Phase 1: LLM이 질문에 대한 옵션 점수를 반환."""
    text, _ = await call_llm(
        model=AGENT_MODELS["selector"],
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"사용자 고민: {user_question}",
        temperature=0.3,
        json_mode=True,
        max_tokens=1200,
    )
    try:
        return extract_json(text)
    except Exception:
        return {
            "is_relevant": False,
            "question_scores": {},
            "redirect_message": "질문 분석 중 오류가 발생했습니다. 다시 시도해주세요.",
        }


# ─── Phase 2: Python computation ─────────────────────────────────────────────

def _compute_individual_score(thinker: dict, question_scores: dict) -> float:
    """개별 사상가 점수 = 4개 카테고리 가중합."""
    def dot_binary(q_scores: dict, thinker_set: set) -> float:
        """binary match: thinker_set에 있으면 1.0, 없으면 0."""
        total_q = sum(q_scores.values())
        if not total_q:
            return 0.0
        return sum(q * (1.0 if opt in thinker_set else 0.0) for opt, q in q_scores.items()) / total_q

    def dot_weighted(q_scores: dict, thinker_weights: dict) -> float:
        """weighted match: thinker_pool.yaml에 명시된 weight 사용."""
        total_q = sum(q_scores.values())
        if not total_q:
            return 0.0
        return sum(q * thinker_weights.get(opt, {}).get("weight", 0.0)
                   for opt, q in q_scores.items()) / total_q

    # primary_domains: binary
    thinker_domains = {d["id"] for d in thinker.get("primary_domains", [])}
    pd_score = dot_binary(question_scores.get("primary_domains", {}), thinker_domains)

    # strongest_question_strengths: weighted
    qs_score = dot_weighted(
        question_scores.get("strongest_question_strengths", {}),
        thinker.get("strongest_question_strengths", {}),
    )

    # strongest_axis_weights: weighted
    ax_score = dot_weighted(
        question_scores.get("strongest_axis_weights", {}),
        thinker.get("strongest_axis_weights", {}),
    )

    # core_concepts: binary
    thinker_concepts = {c["id"] for c in thinker.get("core_concepts", [])}
    cc_score = dot_binary(question_scores.get("core_concepts", {}), thinker_concepts)

    return 0.10 * pd_score + 0.30 * qs_score + 0.45 * ax_score + 0.15 * cc_score


def _compute_coverage_score(group: list[dict], axis_scores: dict) -> float:
    """
    Coverage score for a group of 3 thinkers.
    Uses top-6 axes where question score >= 0.50.
    """
    targets = sorted(
        [(ax, sc) for ax, sc in axis_scores.items() if sc >= 0.50],
        key=lambda x: -x[1],
    )[:6]

    if not targets:
        # fall back: use top-3 axes regardless of threshold
        targets = sorted(axis_scores.items(), key=lambda x: -x[1])[:3]
    if not targets:
        return 0.0

    total_q = sum(sc for _, sc in targets)
    coverage = 0.0
    for ax, q in targets:
        max_weight = max(
            t.get("strongest_axis_weights", {}).get(ax, {}).get("weight", 0.0)
            for t in group
        )
        coverage += q * max_weight

    return coverage / total_q


def _select_best_group(question_scores: dict) -> tuple[list[dict], dict[str, float], list[dict]]:
    """
    C(10,3) 열거로 최적 그룹 선택.
    반환: (best_group_list, individual_scores_dict, top3_groups_info)
    """
    individual_scores = {
        t["thinker_id"]: _compute_individual_score(t, question_scores)
        for t in _THINKERS
    }

    axis_scores = question_scores.get("strongest_axis_weights", {})

    scored: list[tuple] = []
    for combo in combinations(range(len(_THINKERS)), 3):
        group = [_THINKERS[i] for i in combo]
        avg_individual = sum(individual_scores[t["thinker_id"]] for t in group) / 3
        coverage = _compute_coverage_score(group, axis_scores)
        group_score = 0.65 * avg_individual + 0.35 * coverage
        scored.append((group_score, avg_individual, coverage, group))

    scored.sort(key=lambda x: -x[0])

    top3 = []
    for gs, avg_ind, cov, grp in scored[:3]:
        top3.append({
            "thinker_ids": [t["thinker_id"] for t in grp],
            "name_ko":     [t["name_ko"]     for t in grp],
            "group_score":          round(gs,      4),
            "avg_individual_score": round(avg_ind, 4),
            "coverage_score":       round(cov,     4),
            "individual_scores": {
                t["thinker_id"]: round(individual_scores[t["thinker_id"]], 4)
                for t in grp
            },
        })

    best_group = scored[0][3]
    return best_group, individual_scores, top3


# ─── Public interface ────────────────────────────────────────────────────────

async def run_selector(user_question: str) -> dict:
    """
    Phase 1 (LLM 질문 스코어링) + Phase 2 (Python 최적 그룹 선택).

    반환:
    {
        "is_relevant": bool,
        "question_scores": dict,
        "selected_group": [thinker_dict, ...],   # YAML raw dicts, 3개
        "individual_scores": {thinker_id: float},
        "group_score": float,
        "redirect_message": str,
    }
    """
    scored = await _score_question(user_question)

    if not scored.get("is_relevant", False):
        return {
            "is_relevant": False,
            "question_scores": {},
            "selected_group": [],
            "individual_scores": {},
            "group_score": 0.0,
            "redirect_message": scored.get(
                "redirect_message",
                "조금 더 구체적인 인생 고민이나 철학적 질문을 말씀해주시겠어요?"
            ),
        }

    question_scores = scored.get("question_scores", {})
    best_group, individual_scores, top3 = _select_best_group(question_scores)

    return {
        "is_relevant": True,
        "question_scores": question_scores,
        "selected_group": best_group,
        "individual_scores": individual_scores,
        "group_score": top3[0]["group_score"] if top3 else 0.0,
        "top3_groups": top3,
        "redirect_message": "",
    }
