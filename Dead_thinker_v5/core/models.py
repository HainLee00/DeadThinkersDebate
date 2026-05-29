from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThinkerProfile:
    """v5: YAML thinker_pool에서 구성되는 인물 프로필."""
    id: str
    name: str
    name_ko: str
    birth_year: int = 0
    death_year: int = 0
    nationality: str = ""
    domain: str = ""
    identifier: str = ""
    why_selected: str = ""
    expected_stance: str = ""
    assigned_choice: str = ""
    choice_rationale: str = ""
    response_guidance_ko: dict = field(default_factory=dict)


@dataclass
class Message:
    role: str
    speaker: str
    content: str
    round_num: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Round:
    round_num: int
    user_input: str
    gate_result: Optional[dict]
    thinker_responses: list[Message]
    moderator_response: Optional[Message]
    context_summary: Optional[str]


@dataclass
class DebateSession:
    session_id: str
    original_question: str
    selected_thinkers: list[str]
    rounds: list[Round]
    is_ended: bool = False
    evaluation: Optional[dict] = None
