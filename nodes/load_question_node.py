"""
nodes/load_question_node.py
───────────────────────────
LangGraph Node — Load Question

Responsibilities
────────────────
- Pick the next unanswered question from state["questions"]
- Reset per-question counters (retry_count, generated_answer, validation)
"""

from state import FAQAgentState


def load_question_node(state: FAQAgentState) -> FAQAgentState:
    """
    LangGraph node: Picks the next unanswered question from state["questions"] and
    resets per-question counters (retry_count, generated_answer, validation).
    """
    idx      = state["current_index"]
    question = state["questions"][idx]

    print(f"\n{'='*60}")
    print(f"[LoadQuestion] Processing FAQ #{question['faqQuestion']}: {question['Question']}")
    print(f"{'='*60}")

    return {
        **state,
        "current_question": question["Question"],
        "current_faq_id":   question["faqQuestion"],
        "generated_answer": "",
        "validation":       None,
        "retry_count":      0,
        "error":            None,
    }