from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.langchain_matcher import match_string_with_langchain
from app.services.model_factory import get_chat_model


app = FastAPI(
    title="CCS AI Name Matcher",
    version="0.3.0",
    description="Microservice for matching an input string to the best candidate using LLM prompting (mock now, Azure later).",
)

@app.get("/health")
def health():
    return {"status": "ok"}
from fastapi.responses import PlainTextResponse
from fastapi.requests import Request

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(str(exc), status_code=500)

class MatchRequest(BaseModel):
    input_string: str = Field(..., min_length=1, description="The string to match.")
    candidates: List[str] = Field(..., min_items=1, description="Candidate strings to match against.")
    prompt_path: Optional[str] = Field(
        None,
        description="Optional prompt file path. Overrides PROMPT_PATH in env if provided.",
    )

class MatchResponse(BaseModel):
    input_string: str
    match: Optional[str]
    raw: str

def _normalize_output(raw: str) -> Optional[str]:
    s = (raw or "").strip()
    s = s.strip('"').strip("'").strip()
    if s.lower() in {"none", "null", "n/a", "na", ""}:
        return None
    return s


@app.get("/match", response_model=MatchResponse)
def match_get(
    input_string: str = Query(..., min_length=1),
    candidates: List[str] = Query(..., description="Repeat this param for each candidate."),
    prompt_path: Optional[str] = Query(None),
):
    try:
        model = get_chat_model(candidates=candidates)
        raw = match_string_with_langchain(
            input_string=input_string,
            list_of_strings=candidates,
            model=model,
            prompt_path=prompt_path,
        )
        match = _normalize_output(raw)
        return MatchResponse(input_string=input_string, match=match, raw=(raw or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match", response_model=MatchResponse)
def match_post(req: MatchRequest):
    try:
        model = get_chat_model(candidates=req.candidates)
        raw = match_string_with_langchain(
            input_string=req.input_string,
            list_of_strings=req.candidates,
            model=model,
            prompt_path=req.prompt_path,
        )
        match = _normalize_output(raw)
        return MatchResponse(input_string=req.input_string, match=match, raw=(raw or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
