• Partially. rag.py has the core of a RAG pipeline, but it does not meet all of Part 3’s requirements yet.
 
   Requirement                                               Status                What I found                                           
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Load and extract the syllabus PDF                         Partial               It accepts a PDF path from the command line and
                                                                                   extracts text, but doesn’t default to or check for the
                                                                                   required course syllabus.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Split text into overlapping chunks and document the       Partial               It uses a 75-token target and 10-token overlap, but
   settings                                                                        chunks can grow to 112 tokens; the settings aren’t
                                                                                   documented for users.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Embed and store chunks in searchable Chroma               Yes, with a caveat    Chroma stores source text and metadata. No embedding
                                                                                   function is specified, so Chroma uses its default.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Retrieve chunks and answer with gpt-oss                   Partial               It retrieves one chunk and sends it with the question
                                                                                   to gpt-oss.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Reuse a saved index on later runs                         No                    The client uses persistent storage, but each run calls
                                                                                   create_collection, which can fail if the collection
                                                                                   already exists. It also reprocesses the PDF every run.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Say when there isn’t enough information                   Partial               The system prompt asks for this, but retrieval has no
                                                                                   relevance threshold, so it may still answer from a
                                                                                   poor match.
  ────────────────────────────────────────────────────────  ────────────────────  ────────────────────────────────────────────────────────
   Evaluate at least three syllabus questions and report     No                    That’s a separate requirement for rag_eval.py; I don’t
   evidence and ratings                                                            see that file in this project.
