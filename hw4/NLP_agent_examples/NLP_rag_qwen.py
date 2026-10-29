# Qwen RAG for both embedding and LLM

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


##########################################################
# Step 1. Connect to Qwen Generative LLM
##########################################################

llm = ChatOpenAI(
    model="Qwen2.5-72B-Instruct",      # change to your deployed Qwen model name
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://your-qwen-server/v1",  # change to your Qwen endpoint
    temperature=0
)


##########################################################
# Step 2. Load documents
##########################################################

# Option A: Sample documents

documents = [
    Document(
        page_content="""
        HBM demand is increasing because AI accelerators
        require very high memory bandwidth.
        """
    ),

    Document(
        page_content="""
        NAND flash memory is widely used in SSD storage.
        """
    ),

    Document(
        page_content="""
        Samsung, SK hynix and Micron are leading memory
        manufacturers.
        """
    ),
]


# Option B: Load PDF document
# Uncomment if using PDF

pdf_path = "cpsc-handbook-2022-23.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()


##########################################################
# Step 3. Split documents into chunks
##########################################################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)


##########################################################
# Step 4. Create Qwen Embedding Model
##########################################################
# download (~2-4DB) the Qwen embedding model from HuggingFace Hub and load it from cache. 
# The first time you run this, it will take a while to download the model.
embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

##########################################################
# Step 5. Build FAISS vector database
##########################################################

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)


##########################################################
# Step 6. Retrieve relevant documents
##########################################################

question = (
    "Can a student petition to add a course "
    "if a class is closed?"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_docs = retriever.invoke(question)


##########################################################
# Step 7. Build context
##########################################################

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


##########################################################
# Step 8. Prompt template
##########################################################

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say:
"I do not have enough information."

Context:
{context}

Question:
{question}
"""
)


##########################################################
# Step 9. Format prompt
##########################################################

messages = prompt.format_messages(
    context=context,
    question=question
)


##########################################################
# Step 10. Ask Qwen
##########################################################

response = llm.invoke(messages)


##########################################################
# Step 11. Print answer
##########################################################

print(response.content)