import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.services.langchain_matcher import amatch_string_with_langchain
from app.services.model_factory import get_chat_model

app = FastAPI(
    title="CCS AI Name Matcher",
    version="0.4.0",
    description="Microservice for concurrently matching input strings to the best candidates using LLM prompting.",
)


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


@app.post("/match", response_model=MatchResponse)
async def match_post(req: MatchRequest):
    try:
        model = get_chat_model(candidates=req.candidates)
        raw_results = await asyncio.gather(
            *(
                amatch_string_with_langchain(
                    input_string=input_string,
                    list_of_strings=req.candidates,
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
