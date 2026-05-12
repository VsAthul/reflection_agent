"""
main.py
───────
Entry point for the FAQ Reflection Agent.

Workflow
────────
1. Load FAQ questions from inputs/input.json
2. Initialise LangGraph state
3. Run the compiled reflection-agent graph
4. Write results to outputs/output.json
"""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env before any llm imports (so GROQ_API_KEY is available)
load_dotenv()

from graph import faq_graph
from state import FAQAgentState


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

BASE_DIR    = os.path.dirname(__file__)
INPUT_PATH  = os.path.join(BASE_DIR, "inputs",  "input.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "output.json")


def load_questions(path: str) -> list[dict]:
    """Read FAQ questions from the input JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Main] Loaded {len(questions)} questions from {path}")
    return questions


def save_output(output: dict, path: str) -> None:
    """Write the final result dict to the output JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Main] Results saved → {path}")


def build_initial_state(questions: list[dict]) -> FAQAgentState:
    """Create the initial LangGraph state from the question list."""
    return {
        "questions":        questions,
        "current_index":    0,
        "current_question": "",
        "current_faq_id":   0,
        "generated_answer": "",
        "validation":       None,
        "retry_count":      0,
        "results":          [],
        "error":            None,
    }


def run() -> None:
    """Main execution function."""
    print("\n" + "═" * 60)
    print("   FAQ Reflection Agent — Starting")
    print("═" * 60)

    # 1. Load input
    questions = load_questions(INPUT_PATH)

    # 2. Initialise state
    initial_state = build_initial_state(questions)

    # 3. Run LangGraph
    print(f"\n[Main] Running graph for {len(questions)} question(s)...\n")
    final_state = faq_graph.invoke(initial_state)

    # 4. Build output payload
    output = {
        "totalQuestions": len(questions),
        "processedAt":    datetime.now(timezone.utc).isoformat(),
        "results":        final_state["results"],
    }

    # 5. Save output
    save_output(output, OUTPUT_PATH)

    # 6. Print summary
    print("\n" + "═" * 60)
    print("   Summary")
    print("═" * 60)
    for r in final_state["results"]:
        symbol = "✅" if r["status"] == "answered" else "❌"
        print(
            f"  {symbol}  Q{r['faqQuestion']} | "
            f"status={r['status']} | "
            f"iterations={r['iterationCount']}"
        )
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run()