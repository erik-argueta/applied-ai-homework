
import csv
from pathlib import Path

file_path = Path(__file__).resolve().parent / "rag_evaluation_results.csv"

with open(file_path, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
    fieldnames = list(rows[0].keys())

ratings = [
    {
        "accuracy_0_to_2": "2",
        "completeness_0_to_2": "2",
        "relevance_0_to_2": "2",
        "rating_justification": (
            "Correctly identifies grading weights, "
            "letter-grade thresholds, and grading policies."
        ),
        "unsupported_claims_or_omissions": (
            "No major issues identified in the visible evidence."
        ),
    },
    {
        "accuracy_0_to_2": "1",
        "completeness_0_to_2": "1",
        "relevance_0_to_2": "2",
        "rating_justification": (
            "Relevant attendance summary, but specific "
            "participation thresholds and drop conditions "
            "require further verification."
        ),
        "unsupported_claims_or_omissions": (
            "Verify participation-credit thresholds, "
            "make-up rules, and administrative drop policy."
        ),
    },
    {
        "accuracy_0_to_2": "2",
        "completeness_0_to_2": "2",
        "relevance_0_to_2": "2",
        "rating_justification": (
            "Identifies assignment weight, academic integrity, "
            "AI-use expectations, and scheduling limitations."
        ),
        "unsupported_claims_or_omissions": (
            "Confirm the AI-use policy wording against the syllabus."
        ),
    },
]

if len(rows) != len(ratings):
    raise ValueError("Expected exactly three evaluation results.")

for row, rating in zip(rows, ratings):
    row.update(rating)

with open(file_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Preliminary RAG evaluation ratings saved.")
