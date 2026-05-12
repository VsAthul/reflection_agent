import sys
import os
 
# Make root importable when running graph.py directly
sys.path.insert(0, os.path.dirname(__file__))
 
from langgraph.graph import StateGraph, END
 
from state import FAQAgentState
from nodes.generate_node import generate_node
from nodes.validate_node import validate_node
from nodes.load_question_node import load_question_node
from nodes.save_result_node import save_result_node
 
 
MAX_RETRIES = 3  # Maximum generation + validation attempts per question
 
 
# ──────────────────────────────────────────────
# Conditional routing functions
# ──────────────────────────────────────────────
 
def route_after_validation(state: FAQAgentState) -> str:
    from state import ValidationResult
    validation: ValidationResult | None = state.get("validation")
    retry_count = state["retry_count"]
 
    if validation and validation.is_valid:
        print(f"[Router] Answer VALID after {retry_count} attempt(s) → saving result")
        return "save_result"
 
    if retry_count >= MAX_RETRIES:
        print(f"[Router] Max retries ({MAX_RETRIES}) reached → saving with no_answer")
        return "save_result"
 
    print(f"[Router] Answer invalid (attempt {retry_count}/{MAX_RETRIES}) → retrying")
    return "generate_node"
 
 
def route_after_save(state: FAQAgentState) -> str:
    """
    After saving a result, check whether more questions remain.
 
    - "load_question"  → more questions to process
    - END              → all questions done
    """
    if state["current_index"] < len(state["questions"]):
        return "load_question"
    print("\n[Router] All questions processed → ending graph.")
    return END
 
 
 
def build_graph() -> StateGraph:
    """
    Constructs and compiles the FAQ Reflection Agent LangGraph.
 
    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(FAQAgentState)
 
    # ── Register nodes ─────────────────────────
    graph.add_node("load_question",  load_question_node)
    graph.add_node("generate_node",  generate_node)
    graph.add_node("validate_node",  validate_node)
    graph.add_node("save_result",    save_result_node)
 
    # ── Entry point ────────────────────────────
    graph.set_entry_point("load_question")
 
    # ── Static edges ───────────────────────────
    graph.add_edge("load_question", "generate_node")
    graph.add_edge("generate_node", "validate_node")
 
    # ── Conditional edge: after validation ─────
    graph.add_conditional_edges(
        "validate_node",
        route_after_validation,
        {
            "generate_node": "generate_node",  # retry loop
            "save_result":   "save_result",    # valid or exhausted
        },
    )
 
    # ── Conditional edge: after saving result ──
    graph.add_conditional_edges(
        "save_result",
        route_after_save,
        {
            "load_question": "load_question",  # next question
            END:             END,              # finished all questions
        },
    )

    compiled = graph.compile()
    graph_image = compiled.get_graph().draw_mermaid_png()
    with open("graph_image.png","wb") as f:
        f.write(graph_image)
    return compiled
 
 
# ── Singleton compiled graph (imported by main.py) ──
faq_graph = build_graph()