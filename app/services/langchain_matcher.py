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


def _remove_input_from_candidates(input_string: str, candidates: List[str]) -> List[str]:
    """
    Remove the input string from the candidate options (exact match).

    Why: if multiple suppliers use the same *incorrect* buyer name, that incorrect name can
    end up in the options list. If the input string is also present in the options, the LLM
    may "match" the input to itself, preventing a correct match to the canonical buyer name.
    """
    return [c for c in candidates if c != input_string]


def match_string_with_langchain(
    input_string: str,
    list_of_strings: List[str],
    model,
    prompt_path: Optional[str] = None,
) -> str:
    """
    Matches an input string to one of a list of strings using a LangChain model.

    Args:
        input_string: The string to match.
        list_of_strings: A list of strings to match against.
        model: The LangChain model object (e.g., AzureChatOpenAI or mock).
        prompt_path: Optional prompt file path. If not provided, uses Settings.prompt_path.

    Returns:
        The model's response content.
    """
    settings = get_settings()
    effective_prompt_path = (prompt_path or settings.prompt_path or "").strip()

    filtered_candidates = _remove_input_from_candidates(input_string, list_of_strings)

    if effective_prompt_path:
        prompt_template = _load_prompt_text(effective_prompt_path)

        # Prompt file should contain {input_name} and {candidates}
        system_prompt = prompt_template.format(
            input_name=input_string,
            candidates=json.dumps(filtered_candidates, ensure_ascii=False),
        )
    else:
        system_prompt = (
            f"Match the input string to one of these : {filtered_candidates}. "
            f"If you can't find a match, return 'None'."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_string),
    ]

    logger.info("Using LLM to find match for %s", input_string)
    response = model.invoke(messages)

    content = getattr(response, "content", "")
    logger.info("Response = %s", content)
    return content
