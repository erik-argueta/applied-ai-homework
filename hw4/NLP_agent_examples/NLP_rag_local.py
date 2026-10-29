# Using Hugging Face RAG (Retrieval Augmented Generation) example using local LLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter


############################################################
# Step 1. Create local Hugging Face LLM
############################################################

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    do_sample=True,
)

llm = HuggingFacePipeline(
    pipeline=text_generation_pipeline
)

############################################################
# Step 2. Create sample documents
############################################################

documents = [

    Document(
        page_content="""
HBM demand is increasing because AI accelerators
require extremely high memory bandwidth.
"""
    ),

    Document(
        page_content="""
NAND flash memory is widely used for SSD storage.
"""
    ),

    Document(
        page_content="""
Samsung, SK hynix, and Micron are major memory
manufacturers.
"""
    ),

]

############################################################
# Step 3. Split documents
############################################################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

############################################################
# Step 4. Embedding model
############################################################

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############################################################
# Step 5. Build FAISS vector store
############################################################

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model,
)

############################################################
# Step 6. Create Retriever
############################################################

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

############################################################
# Step 7. User question
############################################################

question = "What drives HBM demand?"

############################################################
# Step 8. Retrieve documents
############################################################

retrieved_docs = retriever.invoke(question)

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

############################################################
# Step 9. Prompt
############################################################

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

formatted_prompt = prompt.format(
    context=context,
    question=question,
)

############################################################
# Step 10. Generate answer locally
############################################################

response = llm.invoke(formatted_prompt)

############################################################
# Step 11. Print answer
############################################################

print(response)