import httpx

import os

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Build the prompt that gets sent to the LLM.

    The context chunks retrieved from Qdrant are injected here —
    this is the 'augmented' part of Retrieval Augmented Generation.
    The model is instructed to answer only from the provided context
    to reduce hallucination.
    """
    context = "\n\n".join(
        f"[Chunk {i + 1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    return f"""You are a helpful assistant. Answer the question below using only the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Send the prompt to Ollama and return the generated answer.

    Uses stream=False so we wait for the complete response before returning.
    For large responses, streaming would be better — but keep it simple for now.
    """
    prompt = build_prompt(question, context_chunks)

    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,  # wait for full response
        },
        timeout=120.0,  # Ollama can be slow on CPU — give it time
    )

    response.raise_for_status()
    return response.json()["response"]
