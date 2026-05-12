"""
nodes/generate_node.py
──────────────────────
LangGraph Node 1 — Answer Generation (Groq)

Responsibilities
────────────────
- Receive the current FAQ question from the shared state
- Send it to the Groq LLM (llama-3.3-70b-versatile)
- Parse the structured GeneratedAnswer response
- Write the answer back into the shared state
"""

import traceback
from langchain_core.prompts import ChatPromptTemplate

from state import FAQAgentState, GeneratedAnswer
from llm import groq_llm


# ──────────────────────────────────────────────
# Prompt template for generation
# ──────────────────────────────────────────────

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a knowledgeable technical assistant specializing in AI, "
                "machine learning, and software engineering. "
                "Your task is to answer FAQ questions accurately and clearly. "
                "Keep answers concise (3–6 sentences) yet complete. "
                "Do NOT include filler phrases like 'Great question!'. "
                "Respond with factual, well-structured information only."
            ),
        ),
        (
            "human",
            (
                "Please answer the following FAQ question:\n\n"
                "Question: {question}\n\n"
                "Provide a clear, accurate, and helpful answer."
            ),
        ),
    ]
)

# Bind structured output schema so LangChain parses the response automatically
generation_chain = GENERATION_PROMPT | groq_llm.with_structured_output(GeneratedAnswer)


# ──────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────

def generate_node(state: FAQAgentState) -> FAQAgentState:
    """
    LangGraph node: Generate an answer for the current FAQ question using Groq.

    Updates
    ───────
    - state["generated_answer"] : the answer text produced by Groq
    - state["error"]            : set to an error message if generation fails
    """

    question = state["current_question"]
    retry_count = state["retry_count"]

    print(
        f"\n[GenerateNode] Q{state['current_faq_id']} | Attempt {retry_count + 1}/3 "
        f"| Question: {question}"
    )

    try:
        # Invoke the structured generation chain
        result: GeneratedAnswer = generation_chain.invoke({"question": question})

        print(
            f"[GenerateNode] Generated answer (confidence={result.confidence:.2f}): "
            f"{result.answer[:80]}..."
        )

        return {
            **state,
            "generated_answer": result.answer,
            "error": None,
        }

    except Exception as exc:
        error_msg = f"Generation failed: {exc}\n{traceback.format_exc()}"
        print(f"[GenerateNode] ERROR — {error_msg}")

        return {
            **state,
            "generated_answer": "",
            "error": error_msg,
        }