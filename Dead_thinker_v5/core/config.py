AGENT_MODELS = {
    "selector":           "gpt-4.1-mini",   # 질문 스코어링 — 구조화된 출력
    "stance_coordinator": "gpt-4.1-mini",   # 딜레마 프레임 + 입장 배정
    "thinker":            "gpt-4.1",        # 빙의 발화 — 풍부한 지식 필요
    "moderator":          "gpt-4.1-mini",
    "evaluator":          "gpt-4.1-mini",
    "summarizer":         "gpt-4.1-nano",
    "topic_checker":      "gpt-4.1-nano",
}

MAX_SELECTOR_RETRIES = 1
MAX_ROUNDS = 10
