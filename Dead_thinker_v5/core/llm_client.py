import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """LLM 호출. (response_text, metadata) 반환"""
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = await client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    metadata = {
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
    return text, metadata


def extract_json(text: str) -> dict:
    """텍스트에서 JSON 추출. 코드블록(```json...```) 포함 처리."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if match:
        stripped = match.group(1).strip()
    return json.loads(stripped)
