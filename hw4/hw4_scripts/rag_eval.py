
import csv
import os
from pathlib import Path

import chromadb
from openai import OpenAI


# File locations
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag"
DATABASE_DIR = RAG_DIR / "chroma_test"
OUTPUT_FILE = BASE_DIR / "rag_evaluation_results.csv"

MODEL = "gpt-oss"

QUESTIONS = [
    "What is the grading policy for this course?",
    "What is the attendance policy for this course?",
    "What are the assignment requirements for this course?",
]


def main():
    print("Starting Part 3 RAG Evaluation...\n")

    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not found. Run rag.py first."
        )

    api_key = os.getenv("NRP_API_KEY")

    if not api_key:
        raise RuntimeError("NRP_API_KEY is not configured.")

    # Reuse the saved Chroma database
    db = chromadb.PersistentClient(path=str(DATABASE_DIR))
    collection = db.get_collection(name="syllabus")

    if collection.count() == 0:
        raise RuntimeError("The syllabus collection is empty.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://ellm.nrp-nautilus.io/v1",
    )

    results = []

    for number, question in enumerate(QUESTIONS, start=1):
        print("=" * 60)
        print(f"QUESTION {number}")
        print(question)

        # Retrieve supporting syllabus chunks
        retrieved = collection.query(
            query_texts=[question],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas"],
        )

        documents = retrieved["documents"][0]
        metadata = retrieved["metadatas"][0]

        context = "\n\n".join(documents)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided syllabus "
                        "context. Do not invent information. "
                        "If the context does not contain the answer, "
                        "say that the syllabus does not provide "
                        "enough information."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Syllabus context:\n{context}"
                    ),
                },
            ],
        )

        answer = response.choices[0].message.content or ""

        print("\nGENERATED ANSWER:")
        print(answer)

        print("\nSUPPORTING SYLLABUS TEXT:")

        for index, document in enumerate(documents, start=1):
            print(f"\nSource {index}:")
            print(document[:700])

        # Human evaluation is completed after reviewing
        # the actual syllabus evidence.
        results.append({
            "question": question,
            "generated_answer": answer,
            "supporting_text": context,
            "source": str(metadata),
            "accuracy_0_to_2": "",
            "completeness_0_to_2": "",
            "relevance_0_to_2": "",
            "rating_justification": "",
            "unsupported_claims_or_omissions": "",
        })

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("RAG EVALUATION COMPLETE")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(
        "Review the syllabus and fill in the "
        "accuracy, completeness, and relevance ratings."
    )


if __name__ == "__main__":
    main()
