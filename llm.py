"""
llm.py
──────
LLM configuration for the FAQ Reflection Agent.

Exports two ready-to-use model instances:
- groq_llm   : Groq-hosted LLaMA 3.3-70b (answer generation)
- ollama_llm : Local Ollama LLaMA 3.1-8b  (answer validation)

Both are wrapped with `.with_structured_output()` where needed inside
the node files — raw models are exported here so node authors can apply
their own output schemas.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Load variables from .env at import time
load_dotenv()


# ──────────────────────────────────────────────
# Groq — Answer Generation LLM
# ──────────────────────────────────────────────

def get_groq_llm() -> ChatGroq:
    """
    Returns a ChatGroq instance configured for FAQ answer generation.

    Model  : llama-3.3-70b-versatile
    Purpose: Generate detailed, accurate answers to FAQ questions.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3,          # Low temp → more factual, deterministic answers
        max_tokens=1024,
    )


# ──────────────────────────────────────────────
# Ollama — Answer Validation LLM
# ──────────────────────────────────────────────

def get_ollama_llm() -> ChatOllama:
    """
    Returns a ChatOllama instance configured for answer validation.

    Model  : llama3.1:8b  (must be pulled locally via `ollama pull llama3.1:8b`)
    Purpose: Validate whether the Groq-generated answer is accurate and relevant.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model="llama3.1:8b",
        base_url=base_url,
        temperature=0.1,          # Very low temp → consistent, strict validation
    )


# ──────────────────────────────────────────────
# Convenience singletons (imported by nodes)
# ──────────────────────────────────────────────

groq_llm   = get_groq_llm()
ollama_llm = get_ollama_llm()