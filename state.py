"""
state.py
────────
Central state definitions for the FAQ Reflection Agent.

Contains:
- Pydantic models for structured LLM outputs
- FAQAgentState: the shared state passed through every LangGraph node
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────────────────────────
# Pydantic Models — Structured LLM Outputs
# ──────────────────────────────────────────────

class GeneratedAnswer(BaseModel):
    """
    Structured output produced by the Groq generation node.
    The LLM is asked to respond strictly in this schema.
    """

    answer: str = Field(
        description="A clear, accurate, and concise answer to the FAQ question."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The model's self-reported confidence in the answer (0.0 – 1.0).",
    )


class ValidationResult(BaseModel):
    """
    Structured output produced by the Ollama validation node.
    The validator judges whether the generated answer is acceptable.
    """

    is_valid: bool = Field(
        description=(
            "True if the answer is factually correct, relevant, and sufficiently detailed. "
            "False otherwise."
        )
    )
    remarks: str = Field(
        description="Brief explanation of why the answer is valid or invalid."
    )
    checked_answer: bool = Field(
        description="Always True — confirms the validator reviewed the answer."
    )


# ──────────────────────────────────────────────
# Per-question result stored in the final output
# ──────────────────────────────────────────────

class QuestionResult(TypedDict):
    faqQuestion: int
    Question: str
    iterationCount: int
    status: str          # "answered" | "no_answer"
    Answer: str          # generated answer text, or "-" when no valid answer found
    checkedAnswer: bool
    remarks: str


# ──────────────────────────────────────────────
# LangGraph Shared State
# ──────────────────────────────────────────────

class FAQAgentState(TypedDict):
    """
    The single mutable state object threaded through every node in the graph.

    Fields
    ──────
    questions        : Full list of FAQ dicts loaded from input.json
    current_index    : Index of the question currently being processed
    current_question : The question text being worked on this cycle
    current_faq_id   : The faqQuestion integer id for the active question
    generated_answer : Latest answer text produced by the Groq node
    validation       : Latest ValidationResult from the Ollama node
    retry_count      : How many generation attempts have been made for the current question
    results          : Accumulated list of QuestionResult dicts (final output)
    error            : Optional error message for debugging
    """

    questions: list[dict[str, Any]]
    current_index: int
    current_question: str
    current_faq_id: int
    generated_answer: str
    validation: Optional[ValidationResult]
    retry_count: int
    results: list[QuestionResult]
    error: Optional[str]