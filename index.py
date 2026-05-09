from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 1. Load PDF
loader = PyPDFLoader("data/test.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# 5. Use Ollama (Docker running on localhost:11434)
llm = Ollama(
    model="llama3",  # make sure you pulled this
    base_url="http://localhost:11434"
)

print("✅ PDF processed and stored in vector DB")

# 6. Query loop
while True:
    query = input("\nAsk something (or type 'exit'): ")
    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\n🤖 Answer:")
    print(response)