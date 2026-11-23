"""Ask one syllabus question through the saved Chroma RAG application.

Unlike ``rag_eval.py``, this script handles a single interactive or
command-line question. It reuses the index created by ``rag/rag.py`` and does
not run evaluation questions, compute ratings, or write evaluation results.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "rag" / "chroma_test"
MODEL = "gpt-oss"
BASE_URL = "https://ellm.nrp-nautilus.io/v1"


def answer_question(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Retrieve syllabus evidence and return the model answer and evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Chroma index not found at {DATABASE_DIR}. Run rag/rag.py first."
        )

    api_key = os.getenv("NRP_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NRP_API_KEY (or OPENAI_API_KEY) for Nautilus access.")

    database = chromadb.PersistentClient(path=str(DATABASE_DIR))
    try:
        collection = database.get_collection(name="syllabus")
    except Exception as exc:
        raise RuntimeError(
            "The saved Chroma index does not contain a 'syllabus' collection. "
            "Run rag/rag.py first."
        ) from exc

    count = collection.count()
    if count == 0:
        raise RuntimeError("The saved syllabus collection is empty. Run rag/rag.py first.")

    retrieved = collection.query(
        query_texts=[question.strip()],
        n_results=min(top_k, count),
        include=["documents", "metadatas"],
    )
    documents = retrieved["documents"][0]
    context = "\n\n".join(documents)
    if not context.strip():
        raise RuntimeError("No syllabus evidence was retrieved for this question.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("NAUTILUS_BASE_URL", BASE_URL),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied syllabus context. "
                    "If the context does not provide enough information, say so. "
                    "Do not invent syllabus details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question.strip()}\n\n"
                    f"Retrieved syllabus context:\n{context}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    return answer.strip(), documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask one question about the syllabus using the saved RAG index."
    )
    parser.add_argument(
        "--question",
        help="question to answer; if omitted, the script prompts interactively",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="number of relevant syllabus chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="also print the retrieved syllabus excerpts",
    )
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Question: ")

    try:
        answer, evidence = answer_question(question, top_k=args.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"RAG error: {exc}", file=sys.stderr)
        return 2

    print(f"Question: {question.strip()}")
    print(f"Answer: {answer}")
    if args.show_context:
        print("\nRetrieved syllabus context:")
        for index, excerpt in enumerate(evidence, start=1):
            print(f"\n[{index}] {excerpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
