"""backend/api/routes_onboarding.py
Onboarding endpoints (Phase 3, PRD §8, FR17/FR18).

- POST /onboarding/import: rejects payloads over settings.ONBOARDING_MAX_CHARS with 413 BEFORE
  any service or LLM call; otherwise delegates to parse_import.
- POST /onboarding/questionnaire: per-answer limit is enforced by the request schema (1000 chars).
- GET /onboarding/prompt: returns the copy-paste export prompt and the questionnaire questions.

Both parse endpoints return the OnboardingResult unchanged (candidates grouped, ids,
duplicates_skipped, dropped, suggested_timezone, errors). suggested_timezone is returned only and
is NEVER applied to the User row. The user id always comes from the identity seam.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import CurrentUser, get_current_user
from backend.api.schemas import (
    OnboardingImportRequest,
    OnboardingPromptResponse,
    OnboardingQuestionnaireRequest,
    QuestionOut,
)
from backend.config.settings import settings
from backend.services.onboarding import (
    IMPORT_EXPORT_PROMPT,
    QUESTIONNAIRE,
    OnboardingResult,
    parse_import,
    parse_questionnaire,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onboarding"])


@router.post("/onboarding/import", response_model=OnboardingResult)
async def onboarding_import(
    request: OnboardingImportRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> OnboardingResult:
    # Over the limit -> 413 with no service/LLM call at all (checked before delegating).
    if len(request.text) > settings.ONBOARDING_MAX_CHARS:
        raise HTTPException(status_code=413, detail="payload too large")
    return await parse_import(current_user.user_id, request.text)


@router.post("/onboarding/questionnaire", response_model=OnboardingResult)
async def onboarding_questionnaire(
    request: OnboardingQuestionnaireRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> OnboardingResult:
    return await parse_questionnaire(current_user.user_id, request.answers)


@router.get("/onboarding/prompt", response_model=OnboardingPromptResponse)
async def onboarding_prompt(
    current_user: CurrentUser = Depends(get_current_user),
) -> OnboardingPromptResponse:
    return OnboardingPromptResponse(
        import_prompt=IMPORT_EXPORT_PROMPT,
        questions=[QuestionOut(id=q.id, text=q.text) for q in QUESTIONNAIRE],
    )
