import asyncio
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.langchain_matcher import amatch_string_with_langchain
from app.services.model_factory import get_chat_model

app = FastAPI(
    title="CCS AI Name Matcher",
    version="0.5.0",
    description="Microservice for batch matching input strings to candidates.",
)

# One batch HTTP request expands into one Azure OpenAI request per input. These
# process-wide gates prevent asyncio.gather from sending the whole batch to Azure
# simultaneously. Use one Uvicorn worker, or divide these limits across workers.
_settings = get_settings()
_llm_semaphore = asyncio.Semaphore(max(1, _settings.llm_max_concurrency))
_llm_start_lock = asyncio.Lock()
_last_llm_start = 0.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(str(exc), status_code=500)


class MatchRequest(BaseModel):
    input_strings: List[str] = Field(
        ...,
        min_length=1,
        description="List of input strings to match against candidates.",
    )
    candidates: List[str] = Field(
        ...,
        min_length=1,
        description="Candidate strings to match against.",
    )
    prompt_path: Optional[str] = Field(
        None,
        description="Optional prompt file path. Overrides PROMPT_PATH in env if provided.",
    )


class SingleMatchResult(BaseModel):
    input_string: str
    match: Optional[str]
    raw: str


class MatchResponse(BaseModel):
    results: List[SingleMatchResult]


def _normalize_output(raw: str) -> Optional[str]:
    s = (raw or "").strip()
    s = s.strip('"').strip("'").strip()
    if s.lower() in {"none", "null", "n/a", "na", ""}:
        return None
    return s


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "rate limit" in text


async def _wait_for_rate_gate() -> None:
    """Space request starts globally, including across simultaneous API calls."""
    global _last_llm_start

    interval = max(0.0, _settings.llm_min_request_interval_seconds)
    async with _llm_start_lock:
        delay = interval - (time.monotonic() - _last_llm_start)
        if delay > 0:
            await asyncio.sleep(delay)
        _last_llm_start = time.monotonic()


async def _match_one(
    input_string: str,
    candidates: List[str],
    model,
    prompt_path: Optional[str],
) -> str:
    # The semaphore controls in-flight calls; the rate gate controls call starts.
    async with _llm_semaphore:
        await _wait_for_rate_gate()
        return await amatch_string_with_langchain(
            input_string=input_string,
            list_of_strings=candidates,
            model=model,
            prompt_path=prompt_path,
        )


@app.post("/match", response_model=MatchResponse)
async def match_post(req: MatchRequest):
    try:
        model = get_chat_model(candidates=req.candidates)
        raw_results = await asyncio.gather(
            *(
                _match_one(
                    input_string=input_string,
                    candidates=req.candidates,
                    model=model,
                    prompt_path=req.prompt_path,
                )
                for input_string in req.input_strings
            )
        )

        return MatchResponse(
            results=[
                SingleMatchResult(
                    input_string=input_string,
                    match=_normalize_output(raw),
                    raw=raw or "",
                )
                for input_string, raw in zip(req.input_strings, raw_results)
            ]
        )
    except Exception as exc:
        # Preserve Azure's retryable status instead of hiding a 429 inside a 500.
        status_code = 429 if _is_rate_limit_error(exc) else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
