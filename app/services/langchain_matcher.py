import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_prompt_text(prompt_path: str) -> str:
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return p.read_text(encoding="utf-8")


def _remove_input_from_candidates(
    input_string: str, candidates: List[str]
) -> List[str]:
    """
    Remove the input string from the candidate options (exact match).

    Why: if multiple suppliers use the same *incorrect* buyer name, that incorrect name can
    end up in the options list. If the input string is also present in the options, the LLM
    may "match" the input to itself, preventing a correct match to the canonical buyer name.
    """
    return [c for c in candidates if c != input_string]


def _build_messages(
    input_string: str,
    list_of_strings: List[str],
    prompt_path: Optional[str],
):
    settings = get_settings()
    effective_prompt_path = (prompt_path or settings.prompt_path or "").strip()
    filtered_candidates = _remove_input_from_candidates(input_string, list_of_strings)

    if effective_prompt_path:
        prompt_template = _load_prompt_text(effective_prompt_path)
        system_prompt = prompt_template.format(
            input_name=input_string,
            candidates=json.dumps(filtered_candidates, ensure_ascii=False),
        )
    else:
        system_prompt = (
            f"Match the input string to one of these : {filtered_candidates}. "
            f"If you can't find a match, return 'None'."
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_string),
    ]


def _response_content(response) -> str:
    content = getattr(response, "content", "")
    logger.info("Response = %s", content)
    return content


def match_string_with_langchain(
    input_string: str,
    list_of_strings: List[str],
    model,
    prompt_path: Optional[str] = None,
) -> str:
    """Match one input string using the model's synchronous interface."""
    messages = _build_messages(input_string, list_of_strings, prompt_path)
    logger.info("Using LLM to find match for %s", input_string)
    return _response_content(model.invoke(messages))


async def amatch_string_with_langchain(
    input_string: str,
    list_of_strings: List[str],
    model,
    prompt_path: Optional[str] = None,
) -> str:
    """Match one input without blocking the event loop.

    Native LangChain async invocation is preferred. Models that only expose ``invoke``
    (including the local mock) are run in a worker thread instead.
    """
    messages = _build_messages(input_string, list_of_strings, prompt_path)
    logger.info("Using LLM to find match for %s", input_string)

    ainvoke = getattr(model, "ainvoke", None)
    if callable(ainvoke):
        response = await ainvoke(messages)
    else:
        response = await asyncio.to_thread(model.invoke, messages)

    return _response_content(response)
