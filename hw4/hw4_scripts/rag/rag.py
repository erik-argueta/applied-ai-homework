import os, sys, pymupdf, pathlib 
import textract
import tiktoken
import chromadb     # per assignment instructions
from openai import OpenAI

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

# Extract chunks
def extract_chunk(document,template_prompt):
    prompt = template_prompt.replace('<document>',document)
    messages = [
            {"role": "system", "content": "You help extract information from documents."},
            {"role": "user", "content": prompt}
            ]
    response = client.chat.completions.create(
            model='gpt-4', 
            messages=messages,
            temperature=0,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    return "1." + response.choices[0].message.content

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

def ask_user(collection):
    query = input("What would you like to know?\n> ")
    results = collection.query(query_texts=[query], n_results=1)
    matches = results["documents"][0]
    context = "\n\n".join(matches)

    return query, context

# =============================================
# MAIN
# =============================================

"""Chroma Initialization"""
client = chromadb.Client()
syllabus_collection = client.create_collection(name="syllabus")


# ==========================================================
# Document Ingestion
# ==========================================================
'''
1. Extract txt and create chunks (PyMuPDF -> Python)
2. Generate embeddings (OpenAI embedding model)
3. Store in ChromaDB (Embeddings + txt + metadata)
'''
fname = sys.argv[1]     # gets document filename 
with pymupdf.open(fname) as doc: 
    text = chr(12).join([page.get_text() for page in doc])

pathlib.Path(fname + ".txt").write_bytes(text.encode())
clean_text = text_extraction(fname)

# Initialise tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
results = []
    
chunks = create_chunks(clean_text, 75, tokenizer, overlap=10)
text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

add_data(syllabus_collection, text_chunks, source=fname)
user_query, context = ask_user(syllabus_collection)


# print(text_chunks)

# for index, chunk in enumerate(text_chunks, start=1):
#     print(f"--- Chunk {index} ---")
#     print(chunk.strip())
#     print()


# ==========================================================
# Question Answering
# ==========================================================
'''
1. Embed the question 
2. Query ChromaDB (Retreive most relevant syllabus chunks)
3. Send retreived content + questin to GPT 
4. Generate syllabus-grounded answer.
'''

client = OpenAI(
        api_key=os.environ.get("NRP_API_KEY"),
        base_url="https://ellm.nrp-nautilus.io/v1",
    )

prompt = f"{user_query}. Use this context for answering: {context}"

response = client.chat.completions.create(
    model="gpt-oss",
    messages=[
        {"role": "system", "content": "You are a course advisor. Use only the infomation given. Do not assume or create information."},
        {"role": "user", "content": prompt}
    ],
)

print(response.choices[0].message.content)