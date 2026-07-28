"""Deterministic, explainable organisation-name matching.

The matcher deliberately resolves only high-confidence lexical cases. Ambiguous or
semantic cases are left to the optional LLM fallback in ``langchain_matcher``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

ACRONYM_STOP_WORDS = {"a", "an", "and", "for", "of", "the", "to"}
LEGAL_SUFFIXES = {
    "limited": "ltd",
    "ltd": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "llp": "llp",
    "plc": "plc",
}
NO_NAME_VALUES = {"", "n a", "na", "none", "null", "not applicable", "unknown"}


@dataclass(frozen=True)
class MatchDecision:
    candidate: Optional[str]
    score: float
    runner_up_score: float
    reason: str


def normalise_name(value: str) -> str:
    """Create a Unicode-, case-, punctuation- and legal-suffix-safe form."""
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    words = re.findall(r"[a-z0-9]+", value)
    return " ".join(LEGAL_SUFFIXES.get(word, word) for word in words)


def acronym(value: str) -> str:
    words = normalise_name(value).split()
    significant = [word for word in words if word not in ACRONYM_STOP_WORDS]
    return "".join(word[0] for word in significant if word)


def _similarity(left: str, right: str) -> float:
    """Blend character order and token-order-independent similarity."""
    direct = SequenceMatcher(None, left, right).ratio() # catches typos
    token_sorted = SequenceMatcher(
        None, " ".join(sorted(left.split())), " ".join(sorted(right.split()))
    ).ratio()
    left_tokens, right_tokens = set(left.split()), set(right.split())#gets unique words
    token_dice = (
        2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))
        if left_tokens and right_tokens
        else 0.0
    ) #gets  percentage using Sørensen–Dice Coefficient. measures the overlap or similarity between two sets.
    # Character similarity handles typos; token similarity handles reordered names.
    return max(direct, token_sorted, 0.65 * direct + 0.35 * token_dice)


def deterministic_match(
    input_string: str,
    candidates: Sequence[str],
    *,
    fuzzy_threshold: float = 0.88,
    short_name_threshold: float = 0.84,
    minimum_margin: float = 0.03,
) -> MatchDecision:
    """Return a candidate only when deterministic evidence is strong.

    A confidence margin prevents choosing arbitrarily between similarly close names.
    The original candidate value is always returned, never a generated value.
    """
    source = normalise_name(input_string)
    if source in NO_NAME_VALUES:
        return MatchDecision(None, 0.0, 0.0, "empty_or_placeholder")

    prepared = [(normalise_name(candidate), candidate) for candidate in candidates]
    prepared = [(value, candidate) for value, candidate in prepared if value]
    if not prepared:
        return MatchDecision(None, 0.0, 0.0, "no_candidates")

    # Prefer an exact literal match. This remains deterministic even if another
    # candidate normalises to the same value (for example Ltd/Limited duplicates).
    literal = [candidate for _, candidate in prepared if candidate.casefold().strip() == input_string.casefold().strip()]
    if literal:
        return MatchDecision(literal[0], 1.0, 0.0, "literal_exact")

    normalised_exact = [candidate for value, candidate in prepared if value == source]
    if len(normalised_exact) == 1:
        return MatchDecision(normalised_exact[0], 1.0, 0.0, "normalised_exact")
    if len(normalised_exact) > 1:
        return MatchDecision(None, 1.0, 1.0, "ambiguous_normalised_exact")

    compact_source = source.replace(" ", "")
    if compact_source.isalnum() and 2 <= len(compact_source) <= 10:
        acronym_matches = [
            candidate
            for value, candidate in prepared
            if len(value.split()) >= 2 and acronym(value) == compact_source
        ]
        if len(acronym_matches) == 1:
            return MatchDecision(acronym_matches[0], 1.0, 0.0, "acronym")
        if len(acronym_matches) > 1:
            return MatchDecision(None, 1.0, 1.0, "ambiguous_acronym")

    scores = sorted(
        ((_similarity(source, value), candidate, value) for value, candidate in prepared),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_candidate, best_value = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    threshold = (
        short_name_threshold
        if max(len(source), len(best_value)) <= 5
        else fuzzy_threshold
    )
    if best_score >= threshold and best_score - runner_up >= minimum_margin:
        return MatchDecision(best_candidate, best_score, runner_up, "high_confidence_fuzzy")

    reason = "ambiguous" if best_score >= threshold else "below_threshold"
    return MatchDecision(None, best_score, runner_up, reason)


def candidate_for_model_output(content: str, candidates: Iterable[str]) -> Optional[str]:
    """Map model output back to exactly one supplied candidate."""
    output = normalise_name(content.strip("\"' `\n\t "))
    if output in NO_NAME_VALUES:
        return None
    matches = [candidate for candidate in candidates if normalise_name(candidate) == output]
    return matches[0] if len(matches) == 1 else None
