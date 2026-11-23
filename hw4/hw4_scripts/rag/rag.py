import argparse
import os
import pathlib
import textract
import tiktoken
import chromadb     # per assignment instructions
from openai import OpenAI

DEFAULT_PERSIST_DIRECTORY = pathlib.Path(__file__).resolve().parent / "chroma_test"

# ====================================================
# Functions
# ====================================================

"""Reads the pdf and extracts text only; AI improved the function"""
def text_extraction(filepath):
    rough_text = textract.process(str(filepath), method="pdfminer").decode(
        "utf-8", errors="replace"
    )
    lines = [" ".join(line.split()) for line in rough_text.splitlines()]
    paragraphs = []
    current = []

    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
# AI assisted with this chunk heavily
def create_chunks(text, n, tokenizer, overlap=0):
    """Yield token-list chunks of at most 1.5 * n tokens.

    When possible, each chunk ends at a sentence or paragraph boundary near
    the target size. ``overlap`` controls how many tokens are repeated between
    adjacent chunks.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= n:
        raise ValueError("overlap must be smaller than n")

    tokens = tokenizer.encode(text)
    i = 0
    min_size = max(1, n // 2)
    max_size = max(n, (3 * n) // 2)

    while i < len(tokens):
        end = min(i + max_size, len(tokens))
        earliest_boundary = min(i + min_size, end)

        # Search backward so a chunk uses the latest nearby natural boundary.
        for j in range(end, earliest_boundary - 1, -1):
            candidate = tokenizer.decode(tokens[i:j]).rstrip(" \t\r")
            if candidate.endswith((".", "!", "?", "\n")):
                break

        # Fall back to the target size when no natural boundary is nearby.
        else:
            j = min(i + n, len(tokens))

        # Keep progress guaranteed even for very small chunk sizes.
        j = max(i + 1, j)
        yield tokens[i:j]
        i = max(i + 1, j - overlap)

# Add document chunks and their source information to ChromaDB.
def add_data(collection, chunks, source=None):
    documents = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not documents:
        return

    source_name = pathlib.Path(source).name if source else "document"
    collection.add(
        ids=[f"{source_name}-{index}" for index in range(len(documents))],
        documents=documents,
        metadatas=[
            {"source": str(source) if source else source_name, "chunk_index": index}
            for index in range(len(documents))
        ],
    )

def retrieve_context(collection, query, n_results=1):
    """Return the most relevant stored text for a question."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("question must be a non-empty string")
    count = collection.count()
    if count == 0:
        raise RuntimeError("The RAG collection has no indexed document chunks.")

    results = collection.query(
        query_texts=[query.strip()], n_results=min(n_results, count)
    )
    matches = (results.get("documents") or [[]])[0]
    if not matches:
        return ""
    return "\n\n".join(matches)


def get_collection(document_path, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """Load the persistent syllabus index, creating it from the PDF if empty."""
    document = pathlib.Path(document_path)
    if not document.is_file():
        raise FileNotFoundError(f"Syllabus PDF not found: {document}")

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(name="syllabus")
    if collection.count() == 0:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        clean_text = text_extraction(document)
        chunks = create_chunks(clean_text, 200, tokenizer, overlap=40)
        text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
        add_data(collection, text_chunks, source=document)
    return collection


def answer_question(
    question, document_path, persist_directory=DEFAULT_PERSIST_DIRECTORY
):
    """Retrieve syllabus context and answer a question with Nautilus gpt-oss.

    This function is shared by the typed RAG CLI and ``stt_rag.py`` so a
    transcribed question can enter the same retrieval and answer pipeline.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    api_key = os.environ.get("NRP_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NRP_API_KEY (or OPENAI_API_KEY) for Nautilus access.")

    collection = get_collection(document_path, persist_directory)
    context = retrieve_context(collection, question)
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "NAUTILUS_BASE_URL", "https://ellm.nrp-nautilus.io/v1"
        ),
    )
    response = client.chat.completions.create(
        model="gpt-oss",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied syllabus context. "
                    "If it does not contain enough information, say so. Do not "
                    "invent syllabus details."
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
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the syllabus.")
    parser.add_argument("syllabus_pdf", help="path to the syllabus PDF")
    parser.add_argument(
        "--question", help="question to answer; prompts interactively if omitted"
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_PERSIST_DIRECTORY),
        help="directory for the persistent Chroma index",
    )
    args = parser.parse_args()
    question = args.question or input("What would you like to know?\n> ")
    answer = answer_question(question, args.syllabus_pdf, args.persist_directory)
    print(answer)


if __name__ == "__main__":
    main()
