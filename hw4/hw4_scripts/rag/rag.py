import os, sys, pymupdf, pathlib
import chromadb     # per assignment instructions
from openai import OpenAI


chroma_client = chromadb.Client()

# Creating a collection - where embeddings, docs, and addnl metadata will be stored.
collection = chroma_client.create_collection(name="course_syllabus")


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

# ==========================================================
# Question Answering
# ==========================================================
'''
1. Embed the question 
2. Query ChromaDB (Retreive most relevant syllabus chunks)
3. Send retreived content + questin to GPT 
4. Generate syllabus-grounded answer.
'''



def ai_func(): 
    client = OpenAI(
        api_key=os.environ.get("NRP_API_KEY"),
        base_url="https://ellm.nrp-nautilus.io/v1",
    )

    response = client.chat.completions.create(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": "You are running as if right-out-of-the-box."},
            {"role": "user", "content": "Describe the color green as if I were blind."}
        ],
    )
