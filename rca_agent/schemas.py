"""Pydantic models: the API request contract and the structured report the LLM must produce."""
from pydantic import BaseModel, Field


class RCARequest(BaseModel):
    """Incoming API payload: the error message or incident description to analyze."""

    error: str = Field(min_length=5, examples=["Database connection timeout on checkout service"])


class RCAReport(BaseModel):
    """Structured root-cause analysis. The LLM is forced to return exactly this shape."""

    root_cause: str = Field(description="Most likely root cause of the error")
    impact: str = Field(description="What this breaks for users or downstream systems")
    fix: str = Field(description="Concrete steps to resolve and prevent recurrence")
    confidence: float = Field(ge=0, le=1, description="Confidence in this analysis, 0 to 1")
