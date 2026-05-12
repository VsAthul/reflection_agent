"""
nodes/validate_node.py
──────────────────────
LangGraph Node 2 — Answer Validation (Ollama)

Responsibilities
────────────────
- Receive the generated answer from the shared state
- Ask the local Ollama model (llama3.1:8b) to judge the answer quality
- Parse the structured ValidationResult response
- Write validation results back into state
- Increment retry_count for each validation attempt
"""

import traceback
from langchain_core.prompts import ChatPromptTemplate

from state import FAQAgentState, ValidationResult
from llm import ollama_llm


# ──────────────────────────────────────────────
# Prompt template for validation
# ──────────────────────────────────────────────

VALIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict quality-assurance reviewer for technical FAQ answers. "
                "Your job is to evaluate whether a given answer is:\n"
                "  1. Factually correct\n"
                "  2. Relevant to the question asked\n"
                "  3. Sufficiently detailed (at least 2–3 informative sentences)\n"
                "  4. Free from hallucinations or nonsensical content\n\n"
                "Be strict but fair. An answer that is mostly correct and relevant "
                "should be marked valid. Only mark invalid if it is clearly wrong, "
                "irrelevant, or empty."
            ),
        ),
        (
            "human",
            (
                "Please validate the following FAQ answer.\n\n"
                "Question : {question}\n"
                "Answer   : {answer}\n\n"
                "Is this answer valid? Provide your judgment and a brief remark."
            ),
        ),
    ]
)

# Bind structured output schema
validation_chain = VALIDATION_PROMPT | ollama_llm.with_structured_output(ValidationResult)


# ──────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────

def validate_node(state: FAQAgentState) -> FAQAgentState:
    """
    LangGraph node: Validate the generated answer using the local Ollama model.

    Updates
    ───────
    - state["validation"]   : ValidationResult with is_valid, remarks, checked_answer
    - state["retry_count"]  : incremented by 1 after each validation attempt
    - state["error"]        : set to an error message if validation fails
    """

    question = state["current_question"]
    answer   = state["generated_answer"]
    new_retry_count = state["retry_count"] + 1

    print(
        f"[ValidateNode] Q{state['current_faq_id']} | Attempt {new_retry_count}/3 "
        f"| Validating answer..."
    )

    # If generation itself failed (empty answer), mark invalid immediately
    if not answer:
        fallback = ValidationResult(
            is_valid=False,
            remarks="No answer was generated; skipping validation.",
            checked_answer=True,
        )
        print("[ValidateNode] Skipped — no answer to validate.")
        return {
            **state,
            "validation": fallback,
            "retry_count": new_retry_count,
        }

    try:
        result: ValidationResult = validation_chain.invoke(
            {"question": question, "answer": answer}
        )

        status_tag = "✅ VALID" if result.is_valid else "❌ INVALID"
        print(f"[ValidateNode] {status_tag} | Remarks: {result.remarks}")

        return {
            **state,
            "validation": result,
            "retry_count": new_retry_count,
            "error": None,
        }

    except Exception as exc:
        error_msg = f"Validation failed: {exc}\n{traceback.format_exc()}"
        print(f"[ValidateNode] ERROR — {error_msg}")

        # Treat validation errors as invalid so the retry loop continues
        fallback = ValidationResult(
            is_valid=False,
            remarks=f"Validation error: {str(exc)}",
            checked_answer=True,
        )
        return {
            **state,
            "validation": fallback,
            "retry_count": new_retry_count,
            "error": error_msg,
        }